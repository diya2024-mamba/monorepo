import random

def shuffle_choices(choices, answer, seed=42):
    if seed is not None:
        random.seed(seed)
    shuffled_indices = list(range(len(choices)))
    random.shuffle(shuffled_indices)
    shuffled_choices = [choices[i] for i in shuffled_indices]
    
    # string과 int 둘 다 받아야 함
    if isinstance(answer, str):
        original_answer_index = ord(answer) - ord('A') # A -> 0, B -> 1, C -> 2, D -> 3
    elif isinstance(answer, int):
        original_answer_index = answer
    else:
        raise ValueError("Answer must be a string or an integer")
    
    new_answer_index = shuffled_indices.index(original_answer_index)
    new_answer = chr(ord('A') + new_answer_index)
    return shuffled_choices, new_answer

def shuffle_and_format_choices(doc):
    question = doc['question'].strip()
    choices, new_answer = shuffle_choices(doc['choices'], doc['answer'])
    formatted_choices = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)])
    return f"{question}\n{formatted_choices}\nAnswer:"

def get_shuffled_choice_labels(doc):
    return ["A", "B", "C", "D"]

def get_shuffled_target(doc):
    _, new_answer = shuffle_choices(doc['choices'], doc['answer'])
    return new_answer