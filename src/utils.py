
PAD = 0
UNK = 1

class Config:
    batch_size = 128
    max_len = 300

    embedding_size = 512
    hidden_size = 256
    num_layers = 2
    num_attention_units = 128
    prediction_hidden_size = 256
    optimizer = 'Adam'
    gradient_clip = 5.0

    learning_rate     = 0.0003    # [1.0 for sgd, 0.0003 for adam] work well
    dropout_rate = 0.2