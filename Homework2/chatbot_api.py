from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import os
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray
import time
import torchx
from torchx.runner import get_runner
from torchx.specs import AppDef, Role

ray.init()
runner = get_runner()

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider

# Voc class
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count PAD, SOS, EOS tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def sentenceToIndexes(self, sentence):
        return [self.word2index.get(word, EOS_token) for word in sentence.split(' ')]
    
    
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs)
        return F.softmax(attn_energies.t(), dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding_dropout(self.embedding(input_step))
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = F.softmax(self.out(concat_output), dim=1)
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = torch.argmax(decoder_output, dim=1).unsqueeze(0)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)

        return all_tokens

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    indexes_batch = [voc.sentenceToIndexes(sentence)]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)
    lengths = torch.tensor([len(indexes_batch[0])], dtype=torch.long)
    tokens = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    print("Type 'quit' to exit.")
    while True:
        try:
            input_sentence = input("> ")
            if input_sentence.lower() in ['q', 'quit']:
                break
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_words = [x for x in output_words if x not in ['EOS', 'PAD']]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")

def torchx_inference_worker(texts, encoder_state, decoder_state, voc_state):
    """Worker function that loads model states and performs inference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reinitialize the models inside TorchX worker
    embedding = nn.Embedding(voc_state['num_words'], 500).to(device)
    encoder = EncoderRNN(500, embedding, 2, 0.1).to(device)
    decoder = LuongAttnDecoderRNN('dot', embedding, 500, voc_state['num_words'], 2, 0.1).to(device)

    # Load state dicts into models
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    # Perform inference on texts
    responses = [evaluate(encoder, decoder, searcher, voc_state, text) for text in texts]
    return responses


with open('/Users/bytedance/Documents/COMS_6998/data/save/voc.pkl', 'rb') as f:
    voc = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding = nn.Embedding(voc.num_words, 500)
encoder = EncoderRNN(500, embedding, 2, 0.1).to(device)
decoder = LuongAttnDecoderRNN('dot', embedding, 500, voc.num_words, 2, 0.1).to(device)

checkpoint = torch.load('/Users/bytedance/Documents/COMS_6998/data/save/cb_model_best_model.pth', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)

app = Flask(__name__)

@ray.remote
def remote_generate_response(text):
    try:
        response = evaluate(encoder, decoder, searcher, voc, text)
        return ' '.join(response)
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/batch_chat_ray', methods=['POST'])
def batch_chat_ray():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    start_time = time.time()  # Start timing
    futures = [remote_generate_response.remote(text) for text in data['texts']]
    responses = ray.get(futures)
    total_time = time.time() - start_time  # Calculate elapsed time

    return jsonify({
        'response': responses,
        'time_taken': total_time
    })

@app.route('/torchx_batch_chat', methods=['POST'])
def torchx_batch_chat():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    start_time = time.time()  # Start timing

    # Capture model states and voc state to pass to the worker
    encoder_state = encoder.state_dict()
    decoder_state = decoder.state_dict()
    voc_state = {
        'word2index': voc.word2index,
        'index2word': voc.index2word,
        'num_words': voc.num_words,
    }

    # Define the TorchX role for inference
    role = Role(
        name="inference_worker",
        entrypoint=torchx_inference_worker,
        args=[data['texts'], encoder_state, decoder_state, voc_state],
    )

    app_def = AppDef(name="chatbot_inference", roles=[role])

    # Submit job using Ray scheduler
    job = runner.run(app_def, scheduler="ray")

    # Wait for the job to complete and get the result
    result = runner.wait(job)

    total_time = time.time() - start_time  # Calculate elapsed time

    return jsonify({
        'response': result[0],
        'time_taken': total_time
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    start_time = time.time()  # Start timing
    response = evaluate(encoder, decoder, searcher, voc, data['text'])
    total_time = time.time() - start_time  # Calculate elapsed time

    return jsonify({
        'response': ' '.join(response),
        'time_taken': total_time
    })


@app.route('/batch_chat', methods=['POST'])
def batch_chat():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    start_time = time.time()  # Start timing
    responses = [evaluate(encoder, decoder, searcher, voc, text) for text in data['texts']]
    total_time = time.time() - start_time  # Calculate elapsed time

    return jsonify({
        'response': [' '.join(resp) for resp in responses],
        'time_taken': total_time
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)