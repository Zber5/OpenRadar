from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. calculate the facial parts'score
# 2.

class Autoencoder_Simple(nn.Module):
    def __init__(self):
        super(Autoencoder_Simple, self).__init__()
        in_channel = 12
        out_channel = 16
        kernel_size = 3
        padding = [1, 1]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),

            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.Upsample(scale_factor=(1, 2), mode="bilinear"),

            nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(True),

            nn.Conv2d(out_channel, 11, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.AvgPool2d((1, 5)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        output = torch.squeeze(output)
        return output


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=160):
        return input.view(input.size(0), size, 1, 1)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        in_channel = 12
        out_channel = 16
        kernel_size = 5
        padding = [1, 1]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            # nn.BatchNorm2d(out1_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(out_channel, out_channel * 2, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[1])),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, padding[1])),

            nn.ReLU(inplace=True),

            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(160, 64, kernel_size=(1, kernel_size), stride=1,
                               padding=(0, padding[0])),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, kernel_size), stride=1,
                               padding=(0, padding[0])),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=(1, kernel_size), stride=1,
                               padding=(0, padding[0])),
            nn.ReLU(True),

            nn.ConvTranspose2d(out_channel, 11, kernel_size=(1, kernel_size), stride=1, padding=(0, padding[0])),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(99, 11)
        )

    def forward(self, x):
        x = self.encoder(x)
        output = self.decoder(x)
        output = torch.squeeze(output)
        return output


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class Decoder_CNN(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 device,
                 max_length=100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, conved, encoder_conved):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        # combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(conved_emb, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        # attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = torch.matmul(attention, encoder_conved)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create position tensor
        # pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        # embed tokens and positions
        # tok_embedded = self.tok_embedding(trg)
        # pos_embedded = self.pos_embedding(pos)

        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        # embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        embedded = trg
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)

            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, trg len]

            # calculate attention
            # attention, conved = self.calculate_attention(embedded,
            #                                              conved,
            #                                              encoder_conved,
            #                                              encoder_combined)

            attention, conved = self.calculate_attention(conved,
                                                         encoder_conved)

            # attention = [batch size, trg len, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        # output = [batch size, trg len, output dim]

        return output, attention


class Encoder_CNN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        # self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
        #                                       out_channels=2 * hid_dim,
        #                                       kernel_size=kernel_size,
        #                                       padding=(kernel_size - 1) // 2)
        #                             for _ in range(n_layers)])

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim,
                                              out_channels=2 * input_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [0, 1, 2, 3, ..., src len - 1]

        # pos = [batch size, src len]

        # embed tokens and positions
        # tok_embedded = self.tok_embedding(src)
        # pos_embedded = self.pos_embedding(pos)

        # tok_embedded = pos_embedded = [batch size, src len, emb dim]

        # combine embeddings by elementwise summing
        # embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        # conv_input = self.emb2hid(embedded)
        conv_input = src

        # conv_input = [batch size, src len, hid dim]

        # permute for convolutional layer
        # conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src len]

        # begin convolutional blocks...

        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # ...end convolutional blocks

        # permute and convert back to emb dim
        # conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        # combined = (conved + embedded) * self.scale

        # combined = [batch size, src len, emb dim]

        # return conved, combined
        return conved


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 device,
                 max_length=100):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)

            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, trg len]

            # calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)

            # attention = [batch size, trg len, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        # output = [batch size, trg len, output dim]

        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention


# teacher net structure
class teacherNet(nn.Module):

    def __init__(self):
        super(teacherNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc3(x)
        return x


# student net structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def data_loader():
    return 0


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# train the seq2seq model
if __name__ == "__main__":
    INPUT_DIM = 12
    OUTPUT_DIM = 11
    EMB_DIM = 256
    HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 5  # number of conv. blocks in encoder
    DEC_LAYERS = 5  # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3  # must be odd!
    DEC_KERNEL_SIZE = 3  # can be even or odd
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    TRG_PAD_IDX = 1

    src = torch.zeros((50, 12, 1, 20))
    trg = torch.zeros((50, 11, 1))
    src = src.to(device)
    trg = trg.to(device)

    autoencoder = Autoencoder()
    autoencoder = autoencoder.to(device)

    output = autoencoder(src)

    # Y_train = torch.zeros((50, 11))

    # trg = X_train.permute((1, 0, 2))

    # encoder = Encoder_CNN(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    #
    # encoder = encoder.to(device)
    #
    # decoder = Decoder_CNN(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)
    #
    # encoder_conved = encoder(src)
    #
    # output, attention = decoder(trg, encoder_conved)

    # encoder_conved = [batch size, src len, emb dim]
    # encoder_combined = [batch size, src len, emb dim]

    # calculate predictions of next words
    # output is a batch of predictions for each word in the trg sentence
    # attention a batch of attention scores across the src sentence for
    #  each word in the trg sentence
    # output, attention = self.decoder(trg, encoder_conved, encoder_combined)

    # output = [batch size, trg len - 1, output dim]
    # attention = [batch size, trg len - 1, src len]

    #
    #
    #
    # INPUT_DIM = len(SRC.vocab)
    # OUTPUT_DIM = len(TRG.vocab)
    # EMB_DIM = 256
    # HID_DIM = 512  # each conv. layer has 2 * hid_dim filters
    # ENC_LAYERS = 10  # number of conv. blocks in encoder
    # DEC_LAYERS = 10  # number of conv. blocks in decoder
    # ENC_KERNEL_SIZE = 3  # must be odd!
    # DEC_KERNEL_SIZE = 3  # can be even or odd
    # ENC_DROPOUT = 0.25
    # DEC_DROPOUT = 0.25
    # TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    #
    # enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    # dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)
    #
    # model = Seq2Seq(enc, dec).to(device)
    #
    # optimizer = optim.Adam(model.parameters())
    #
    # criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    #
    # N_EPOCHS = 10
    # CLIP = 0.1
    #
    # best_valid_loss = float('inf')
    #
    # for epoch in range(N_EPOCHS):
    #
    #     start_time = time.time()
    #
    #     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    #     valid_loss = evaluate(model, valid_iterator, criterion)
    #
    #     end_time = time.time()
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), 'tut5-model.pt')
    #
    #     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    #
    #
    #
    # model.load_state_dict(torch.load('tut5-model.pt'))
    #
    # test_loss = evaluate(model, test_iterator, criterion)
    #
    # print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# 1. only match 12 channel's phase information and heatmap


# 2. does heatmap in cv need to be improved


# 3.


# network design
# input : differences phase and energy
# supervision: confidence score / facial action unit
#
