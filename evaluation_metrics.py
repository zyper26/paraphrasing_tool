from evaluate import load

def get_bert_score(predictions, references):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return [round(v, 2) for v in results["f1"]]
