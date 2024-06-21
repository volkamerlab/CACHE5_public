import re

from transformers import T5Tokenizer, T5EncoderModel
import torch


def ProstT5(aa_seqs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
    prostt5.full() if device == "cpu" else prostt5.half()

    seqs = [" ".join(["<fold2AA>"] + list(re.sub("[OUZ]", "X", seq))) for seq in aa_seqs]
    max_len = max(len(s) for s in aa_seqs)
    encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest", return_tensors="pt").to(
        device)
    with torch.no_grad():
        aaseq_embed = prostt5(encoding.input_ids, attention_mask=encoding.attention_mask)
    return dict(zip(aa_seqs, aaseq_embed.last_hidden_state[:, 1: max_len + 1].mean(dim=1).cpu()))


def ProtT5(aa_seqs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_base_mt_uniref50", do_lower_case=False)
    prott5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_base_mt_uniref50").to(device)
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in aa_seqs]

    ids = tokenizer(seqs, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    max_len = max(len(s) for s in seqs)

    with torch.no_grad():
        embedding_repr = prott5(input_ids=input_ids, attention_mask=attention_mask)
    return dict(zip(aa_seqs, embedding_repr.last_hidden_state[:, 1: max_len + 1].mean(dim=1).cpu().numpy()))
