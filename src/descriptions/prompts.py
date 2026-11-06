PROMPT_VERSION = 'v1'


def _domain(dataset):
    domains = {
        'AWA2': ('animal species', 'animal', 'animals', 'species', 'their physical characteristics, habitat, diet, behavior, and any notable distinctions in their roles in the ecosystem', 'its physical characteristics, habitat, diet, behavior, and any notable distinctions in its roles in the ecosystem'),
        'CUB': ('bird species', 'bird', 'birds', 'species', 'physical characteristics such as size, plumage, bill shape and coloration, habitat, diet, behavior, and any notable distinctions in their roles in the ecosystem', 'physical characteristics such as size, plumage, bill shape and coloration, habitat, diet, behavior, and any notable distinctions in its roles in the ecosystem'),
        'SUN': ('scene categories', 'scene', 'scenes', 'categories', 'typical layout and structure, characteristic objects, lighting and materials, function, and the activities that usually take place there', 'typical layout and structure, characteristic objects, lighting and materials, function, and the activities that usually take place there'),
    }
    if dataset not in domains:
        raise ValueError(f'Unknown dataset: {dataset}')
    return domains[dataset]


def pairwise_prompt(dataset, name_a, name_b):
    subject, singular, plural, category, aspects, _ = _domain(dataset)
    system = (f'You are tasked with comparing two {subject} based on their semantic relationships. '
              f'Use your knowledge to generate a detailed and accurate description of the similarities and differences between the given {plural}. '
              f'Ensure the comparison is domain-relevant and includes key characteristics that would help identify or distinguish these {category} in a real-world classification task.')
    user = (f'Describe the similarities and differences between {name_a} and {name_b}. '
            f'Focus on key aspects such as {aspects}. '
            f'Provide a concise but detailed explanation that highlights how these {plural} are alike and how they differ.')
    return system, user


def classwise_prompt(dataset, name):
    subject, singular, _, category, _, aspects = _domain(dataset)
    system = (f'You are tasked with describing one {"scene category" if dataset == "SUN" else subject} based on its characteristics. '
              f'Use your knowledge to generate a detailed and accurate description of the given {singular}. '
              f'Ensure the description is domain-relevant and includes key characteristics that would help identify this {"category" if dataset == "SUN" else "species"} in a real-world classification task.')
    return system, f'Describe the {name}. Focus on key aspects such as {aspects}.'
