
"The implementation of the methods in Prompt class are the modified version of https://github.com/sebastianGehrmann/CausalMediationAnalysis"

from transformers import RobertaTokenizer, BertTokenizer, DistilBertTokenizer
from transformers import  RobertaModel, DistilBertForMaskedLM, RobertaForMaskedLM, BertForMaskedLM

from functools import partial
import pdb

from tqdm import tqdm
import torch
import torch.nn.functional as F

class Prompt():
    '''
    Wrapper for all the possible prompts
    '''
    def __init__(self,
                 tokenizer,
                 base_string: str,
                 substitutes: list,
                 candidates: list,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer


        # All the initial strings
        self.base_strings = [base_string.format(s)
                             for s in substitutes]
        # Tokenized bases
        self.base_strings_tok = [
            self.enc.encode(s,
                            add_special_tokens=False,
                            add_space_before_punct_symbol=True,
                            max_length=10,  # truncate all sentences.
                            pad_to_max_length=True, )
            for s in self.base_strings
        ]

        self.base_strings_tok = torch.LongTensor(self.base_strings_tok)\
                                     .to(device)

        self.position = base_string.split().index('{}')

        self.candidates = []
        for c in candidates:
            # 'a ' added to input so that tokenizer understand that first word follows a space.
            tokens = self.enc.tokenize(
                'a ' + c,
                add_space_before_punct_symbol=True)[1:]
            self.candidates.append(tokens)

        self.candidates_tok = [self.enc.convert_tokens_to_ids(tokens)
                               for tokens in self.candidates]




class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 device='cpu',
                 output_attentions=False,
                 random_weights=False,
                 masking_approach=1,
                 version='bert-base'):
        super()

        self.is_bert = version.startswith('bert')
        self.is_distilbert = version.startswith('distilbert')
        self.is_distilroberta = version.startswith('distilroberta')
        self.is_roberta = version.startswith('roberta')

        # assert (
        #         self.is_bert or self.is_distilbert or self.is_roberta)

        self.device = device
        self.model = (
                      BertForMaskedLM if self.is_bert else
                      DistilBertForMaskedLM if self.is_distilbert else
                      RobertaForMaskedLM if self.is_distilroberta else
                      RobertaForMaskedLM).from_pretrained(
            version,
            output_attentions=output_attentions)
        self.model.eval()
        self.model.to(device)
        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        self.num_layers = self.model.config.num_hidden_layers
        self.masking_approach = masking_approach # Used only for masked LMs
        assert masking_approach in [1, 2, 3, 4, 5, 6]

        tokenizer = (
                     BertTokenizer if self.is_bert else
                     DistilBertTokenizer if self.is_distilbert else
                     RobertaTokenizer if self.is_distilroberta else
                     RobertaTokenizer).from_pretrained(version)
        # Special token id's: (mask, cls, sep)
        self.st_ids = (tokenizer.mask_token_id,
                       tokenizer.cls_token_id,
                       tokenizer.sep_token_id)

        # To account for switched dimensions in model internals:
        # Default: [batch_size, seq_len, hidden_dim],
        self.order_dims = lambda a: a

        if self.is_bert:
            self.word_emb_layer = self.model.bert.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.bert.encoder.layer[layer].output
        elif self.is_distilbert:
            self.word_emb_layer = self.model.distilbert.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.distilbert.transformer.layer[layer].output_layer_norm
        elif self.is_roberta:
            self.word_emb_layer = self.model.roberta.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.roberta.encoder.layer[layer].output
        elif self.is_distilroberta:
            self.word_emb_layer = self.model.roberta.embeddings.word_embeddings
            self.neuron_layer = lambda layer: self.model.roberta.encoder.layer[layer].output



    def get_representations(self, context, position):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):

            representations[layer] = output[self.order_dims((0, position))]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            handles.append(self.word_emb_layer.register_forward_hook(
                partial(extract_representation_hook,
                        position=position,
                        representations=representation,
                        layer=-1)))
            # hidden layers
            for layer in range(self.num_layers):
                handles.append(self.neuron_layer(layer).register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            self.model(context.unsqueeze(0))
            for h in handles:
                h.remove()

        return representation

    def get_probabilities_for_examples(self, context, candidates):
        """Return probabilities of single-token candidates given context"""

        for c in candidates:
            if len(c) > 1:
                raise ValueError(f"Multiple tokens not allowed: {c}")
        outputs = [c[0] for c in candidates]
        logits = self.model(context)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[:, outputs].tolist()



    def experiment(self, word2prompt  ):
        """
        run multiple prompt experiments
        """

        word2prompts_results = {}
        for word in tqdm(word2prompt, desc='words'):
            word2prompts_results[word] = self.single_token_experiment(
                word2prompt[word])


        return word2prompts_results


    def single_token_experiment(self, prompt):
        """
        run one-tokened prompt experiment
        """

        with torch.no_grad():
            '''
            Compute representations for base terms (one for each side of bias)
            '''
            if self.is_bert or self.is_distilbert or self.is_roberta or self.is_distilroberta:
                num_alts = prompt.base_strings_tok.shape[0]

                masks = torch.tensor([self.st_ids[0]]).repeat(num_alts, 1).to(self.device)
                prompt.base_strings_tok = torch.cat(
                    (prompt.base_strings_tok, masks), dim=1)

            base_representations = self.get_representations(
                prompt.base_strings_tok[0],
                prompt.position)

            # Probabilities without prompt (Base case)
            candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
                prompt.base_strings_tok[0].unsqueeze(0),
                prompt.candidates_tok)[0]

        return (candidate1_base_prob, candidate2_base_prob)




