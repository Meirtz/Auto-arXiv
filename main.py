import requests
from bs4 import BeautifulSoup
import openai
import os
from googletrans import Translator
import datetime


class LLMClassifier:
    def __init__(self, api_key=None, model="gpt-4", max_tokens=50, temperature=0):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError("API key must be provided or set as an environment variable")
        openai.api_key = self.api_key
        
        self.configure(model, max_tokens, temperature)
    
    def configure(self, model, max_tokens, temperature):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def classify(self, title):
        prompt = (f"Given the title '{title}', would you categorize this paper "
          f"as belonging to the field of Large Language Models (LLM), "
          f"which includes topics like training large-scale neural networks, "
          f"natural language processing, and language model fine-tuning?")
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        answer = response['choices'][0]['message']['content'].strip()
        print(f"Answer: {answer}")
        is_llm_related = self.parse(answer)
        return is_llm_related, answer  # Return both the classification and the answer
    
    def parse(self, answer):
        # Use GPT-4 to further analyze the answer
        parsing_prompt = (
            f"Based on the statement: '{answer}', determine if it indicates a positive or negative response "
            "to the query about relevance to Large Language Models (LLM) field. Please respond with 'yes' or 'no' only."
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": parsing_prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        parsed_answer = response['choices'][0]['message']['content'].strip().lower()
        print(f"Parsed Answer: {parsed_answer}")
        # check if answer cotainin yes
        return 'yes' in parsed_answer
    
    def translate(self, text, target_language='zh-cn'):
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text


class PageParser:
    def __init__(self):
        pass  # No initialization needed for now
    
    def extract_papers(self, url):
        papers = []
    
        # Send HTTP GET request to fetch the page content
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
    
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
    
        # Find all <dt> elements, each representing a paper entry
        paper_entries = soup.find_all('dt')
    
        # Iterate through each paper entry and extract the title and link
        for entry in paper_entries:
            title_element = entry.find_next('div', class_='list-title')
            title = title_element.text.replace('Title: ', '').strip()
            link_element = entry.find('a', title='Abstract')
            link = 'https://arxiv.org' + link_element['href']
            papers.append({'title': title, 'link': link})
    
        return papers
    
    def extract_abstract(self, url):
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        abstract_block = soup.find('blockquote', class_='abstract')
        if abstract_block:
            abstract_text = abstract_block.text.replace('Abstract:', '').strip()
            return abstract_text
        return None



if __name__ == "__main__":
    classifier = LLMClassifier(model="gpt-3.5-turbo")  # Or set the OPENAI_API_KEY environment variable
    parser = PageParser()
    url = 'https://arxiv.org/list/cs.CL/pastweek?skip=0&show=1000'
    papers = parser.extract_papers(url)
    num_of_papers = len(papers)
    print(f"Total papers: {num_of_papers}")

    llm_related_papers = []

    for i, paper in enumerate(papers):
        print(paper)
        title = paper['title']
        print(f"({i+1}/{num_of_papers})Examining paper: {title}")
        is_llm_related, classification_answer = classifier.classify(title)
        if is_llm_related:
            llm_related_papers.append((paper, classification_answer))

    # file to save today's papers
    today = datetime.date.today()
    output_file = f'llm_related_papers_{today}.txt'

    print(f"Total LLM related papers: {len(llm_related_papers)}")

    for i, (paper, classification_answer) in enumerate(llm_related_papers, 1):
        print(f"{i}. Title: {paper['title']}\n   Link: {paper['link']}\n   Classification Answer: {classification_answer}")
        # Save to file
        with open(output_file, 'a') as f:
            f.write(f"{i}. Title: {paper['title']}\n   Link: {paper['link']}\n   Classification Answer: {classification_answer}\n\n")
        # Extract and translate the abstract
        abstract = parser.extract_abstract(paper['link'])
        if abstract:
            translated_abstract = classifier.translate(abstract)
            print(f"   Abstract: {abstract}\n   Translated Abstract: {translated_abstract}\n")
            # Save to file
            with open(output_file, 'a') as f:
                f.write(f"   Abstract: {abstract}\n\n  Translated Abstract: {translated_abstract}\n\n\n")
        else:
            print("   Abstract not found\n\n\n")