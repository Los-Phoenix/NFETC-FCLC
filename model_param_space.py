param_space_nfetc_wikim = {
    'wpe_dim': 85,
    'lr': 0.0002,
    'state_size': 180,
    'hidden_layers': 0,
    'hidden_size': 0,
    'dense_keep_prob': 0.7,
    'rnn_keep_prob': 0.9,
    'l2_reg_lambda': 0.0000,
    'batch_size': 512,
    'num_epochs': 20,
    'aux_clusters': 3,  # additional cluster number
    'alpha': 0.4,
    'beta': 0.75,  # weight of MLE loss
    'gamma': 0.3,  # weight of KL loss
    'omega': 0.3,  # weight of the pseudo deviation loss
    'delta': 0.1,  # weight of the sharpen loss
    'e_1': 5,  # 1st phase epoch
    'e_2': 12,  # 2nd phase epoch
    'e_3': 10,  # last phase epoch
}

param_space_nfetc_ontonotes_ori = {
    'wpe_dim': 70,
    'hidden_layers': 2,
    'hidden_size': 1024,
    'dense_keep_prob': 0.7,
    'rnn_keep_prob': 0.6,
    'num_epochs': 50,
    'lr': 0.0006,
    'state_size': 1024,
    'l2_reg_lambda': 0.0000,
    'batch_size': 512,
    'alpha': 0.2,
    'beta': 0.8,  # weight of MLE loss
    'gamma': 0.3,  # weight of KL loss
    'omega': 0.3,  # weight of the pseudo deviation loss
    'delta': 0.1,  # weight of the sharpen loss
    'e_1': 8,  # 1st phase epoch
    'e_2': 18,  # 2nd phase epoch
    'e_3': 24,  # last phase epoch
}

param_space_nfetc_ontonotes = {
    'wpe_dim': 70,
    'hidden_layers': 2,
    'hidden_size': 1024,
    'dense_keep_prob': 0.7,
    'rnn_keep_prob': 0.6,
    'num_epochs': 30,
    'lr': 0.0006,
    'state_size': 1024,
    'l2_reg_lambda': 0.0000,
    'batch_size': 512,
    'aux_clusters': 3,  # additional cluster number
    'alpha': 0.2,
    'beta': 0.75,  # weight of the original loss
    'gamma': 0.3,  # weight of KL loss
    'omega': 0.3,  # weight of the pseudo deviation loss
    'delta': 0.1,  # weight of the sharpen loss
    'e_1': 17,  # 1st phase epoch
    'e_2': 53,  # 2nd phase epoch
    'e_3': 12,  # last phase epoch
}

param_space_nfetc_bbn = {
    'lr': 0.0007,  # learning rate
    'state_size': 300,  # LSTM dim
    'l2_reg_lambda': 0.000,  # l2 factor
    'alpha': 0.0,  # control the hier loss
    'wpe_dim': 20,  # position embedding dim
    'hidden_layers': 1,  # number of hidden layer of the classifier
    'hidden_size': 560,  # hidden layer dim of the classifier
    'dense_keep_prob': 0.3,  # dense dropout rate of the feature extractor
    'rnn_dense_dropout': 0.3,  # useless
    'rnn_keep_prob': 1.0,  # rnn output droput rate
    'batch_size': 512,  
    'aux_clusters': 50,  # additional cluster number
    'beta': 0.5,  # weight of MLE loss
    'gamma': 0.1,  # weight of KL loss
    'omega': 0.4,  # weight of the pseudo deviation loss
    'delta': 0.4,  # weight of the sharpen loss
    'num_epochs': 30,
    'e_1': 20,  # 1st phase epoch
    'e_2': 60,  # 2nd phase epoch
    'e_3': 0,  # last phase epoch
}

param_space_dict = {
    'ontonotes': param_space_nfetc_ontonotes,  # the best hp for NFETC-AR in OntoNotes
    'bbn': param_space_nfetc_bbn,  # the best hp for NFETC-AR in BBN
    'wikim': param_space_nfetc_wikim,  # the best hp for NFETC-AR in Wikim
}

int_params = [
    'wpe_dim', 'state_size', 'batch_size', 'num_epochs', 'hidden_size', 'hidden_layers', 'sedim', 'selayer'
]


class ModelParamSpace:
    def __init__(self, learner_name):
        s = 'Wrong learner name!'
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_into_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_into_param(v[i])
                elif isinstance(v, dict):
                    self._convert_into_param(v)
        return param_dict
