import openai
import argparse


def generate_description(class_name_1, class_name_2, model="gpt-4"):
    prompt = f"""
    Describe the similarities and differences between '{class_name_1}' and '{class_name_2}'. 
    Focus on key aspects such as their physical characteristics, habitat, diet, behavior, 
    and any notable distinctions in their roles in the ecosystem. 
    Provide a concise but detailed explanation that highlights how these animals are alike and how they differ.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are tasked with comparing two animal species based on their semantic relationships. "
                           "Use your knowledge to generate a detailed and accurate description of the similarities "
                           "and differences between the given animals. Ensure the comparison is domain-relevant "
                           "and includes key characteristics that would help identify or distinguish these species "
                           "in a real-world classification task."
            },
            {"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class descriptions using an LLM.")
    parser.add_argument("--class_name", type=str, required=True,
                        help="Class name for which to generate the description.")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    args = parser.parse_args()

    description = generate_description(args.class_name, args.model)
    print(f"Generated Description for {args.class_name}:\n{description}")
