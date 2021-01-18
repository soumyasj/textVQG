"""Contains code for the textVQG model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP



class textVQG(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size,
                  sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=True,
                 input_dropout_p=0, dropout_p=0,
                 encoder_max_len=None, num_att_layers=2, att_ff_size=512,
                 embedding=None, z_size=20):
        
        super(textVQG, self).__init__()
        self.hidden_size = hidden_size
        if encoder_max_len is None:
            encoder_max_len = max_len
        self.num_layers = num_layers
        
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.answer_encoder = EncoderRNN(vocab_size, max_len, hidden_size,
                                         input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p,
                                         n_layers=num_layers,
                                         bidirectional=False,
                                         rnn_cell=rnn_cell,
                                         variable_lengths=True)

        
        self.answer_attention = MLP(2*hidden_size, att_ff_size, hidden_size,
                                    num_layers=num_att_layers)
        

        
        self.z_decoder = nn.Linear(z_size, hidden_size)
        self.gen_decoder = MLP(hidden_size, att_ff_size, hidden_size,
                               num_layers=num_att_layers)
        self.decoder = DecoderRNN(vocab_size, max_len, hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  embedding=embedding)

        
        self.mu_answer_encoder = nn.Linear(hidden_size, z_size)
        self.logvar_answer_encoder = nn.Linear(hidden_size, z_size)
         
    

    def encode_images(self, images):
        
        return self.encoder_cnn(images)

    

    def encode_answers(self, answers, alengths):
        
        _, encoder_hidden = self.answer_encoder(
                answers, alengths, None)
        if self.answer_encoder.rnn_cell == nn.LSTM:
            encoder_hidden = encoder_hidden[0]

        
        encoder_hidden = encoder_hidden[-1, :, :].squeeze()
        return encoder_hidden

    def encode_into_z(self, image_features, answer_features):
        
        together = torch.cat((image_features, answer_features), dim=1)
        attended_hiddens = self.answer_attention(together)
        mus = self.mu_answer_encoder(attended_hiddens)
        logvars = self.logvar_answer_encoder(attended_hiddens)
        return mus, logvars

    
    def decode_questions(self, image_features, zs,
                         questions=None, teacher_forcing_ratio=0,
                         decode_function=F.log_softmax):
        
        batch_size = zs.size(0)
        z_hiddens = self.z_decoder(zs)
        if image_features is None:
            hiddens = z_hiddens
        else:
            hiddens = self.gen_decoder(image_features + z_hiddens)

        
        hiddens = hiddens.view((1, batch_size, self.hidden_size))
        hiddens = hiddens.expand((self.num_layers, batch_size,
                                  self.hidden_size)).contiguous()
        if self.decoder.rnn_cell is nn.LSTM:
            hiddens = (hiddens, hiddens)
        result = self.decoder(inputs=questions,
                              encoder_hidden=hiddens,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def forward(self, images, answers, alengths=None, questions=None,
                teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        image_features = self.encode_images(images)
        answer_hiddens = self.encode_answers(answers, alengths)

        
        mus, logvars = self.encode_into_z(image_features, answer_hiddens)
        zs = self.reparameterize(mus, logvars)
        result = self.decode_questions(image_features, zs,
                                       questions=questions,
                                       decode_function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return result

    

    def encode_from_answer(self, images, answers, lengths=None):
        image_features = self.encode_images(images)
        answer_hiddens = self.encode_answers(answers, lengths)
        mus, logvars = self.encode_into_z(image_features, answer_hiddens)
        zs = self.reparameterize(mus, logvars)
        return image_features, zs

    

    def predict_from_answer(self, images, answers, lengths=None,
                            questions=None, teacher_forcing_ratio=0,
                            decode_function=F.log_softmax):
        image_features, zs = self.encode_from_answer(images, answers, lengths=lengths)
        outputs, _, _ = self.decode_questions(image_features, zs, questions=questions,
                                              decode_function=decode_function,
                                              teacher_forcing_ratio=teacher_forcing_ratio)
        return self.parse_outputs_to_tokens(outputs)

    def flatten_parameters(self):
        if hasattr(self, 'decoder'):
            self.decoder.rnn.flatten_parameters()
        if hasattr(self, 'encoder'):
            self.encoder.rnn.flatten_parameters()

    def generator_parameters(self):
        params = self.parameters()
        params = filter(lambda p: p.requires_grad, params)
        return params

    def info_parameters(self):
        params = (list(self.answer_attention.parameters()) +
                  list(self.mu_answer_encoder.parameters()) +
                  list(self.logvar_answer_encoder.parameters()))

        params = filter(lambda p: p.requires_grad, params)
        return params

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def modify_hidden(self, func, hidden, rnn_cell):
        
        if rnn_cell is nn.LSTM:
            return (func(hidden[0]), func(hidden[1]))
        return func(hidden)

    def parse_outputs_to_tokens(self, outputs):
        
        outputs = [o.max(1)[1] for o in outputs]

        outputs = torch.stack(outputs)  
        outputs = outputs.transpose(0, 1) 
        return outputs

    
