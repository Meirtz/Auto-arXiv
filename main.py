import requests
from bs4 import BeautifulSoup
import openai
import os
from googletrans import Translator
import datetime
import time




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
    
    def classify(self, title, abstract, max_retries=10, retry_delay=5):
        prompt = (
            f"Given the title '{title}', and abstract '{abstract}', would you categorize this paper "
            f"as belonging to the field of Large Language Models (LLM), "
            f"which includes topics like training large-scale neural networks, "
            f"natural language processing, and language model fine-tuning? "
            f"If yes, respond with 'yes'. If no, respond with 'no'."
        )
        retries = 0
        while retries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are a helpful assistant tasked to classify titles based on the provided prompt: {prompt}"},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                answer = response['choices'][0]['message']['content'].strip().lower()
                print(f"Answer: {answer}")
                return answer == 'yes', answer  # Return both the classification and the answer
            except openai.error.ServiceUnavailableError:
                print(f"Service unavailable, retrying in {retry_delay} seconds... ({retries+1}/{max_retries})")
                retries += 1
                time.sleep(retry_delay)
        raise Exception(f"Max retries reached. Could not classify title: {title}")




class PageParser:
    def __init__(self):
        self.translator = Translator()


    def translate(self, text, target_language='zh-cn', max_retries=5, retry_delay=2):

        retries = 0

        while retries < max_retries:
            try:
                translated = self.translator.translate(text, dest=target_language)
                return translated.text
            except Exception as e:  # Catching all exceptions, consider specifying the exception types if known.
                print(f"Translation failed: {e}. Retrying {retries + 1}/{max_retries}...")
                retries += 1
                time.sleep(retry_delay)

        print(f"Translation failed after {max_retries} retries. Returning original text.")
        return text  # Return the original text if translation fails after max retries.

    def extract_papers(self, url, max_retries=5, retry_delay=2):
        papers = []
        retries = 0

        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                paper_entries = soup.find_all('dt')

                for entry in paper_entries:
                    meta_div = entry.find_next('dd').find('div', class_='meta')
                    title_element = meta_div.find('div', class_='list-title')
                    title = title_element.text.replace('Title: ', '').strip()
                    authors_element = meta_div.find('div', class_='list-authors')
                    authors = ', '.join(author.text for author in authors_element.find_all('a'))
                    abstract_element = meta_div.find('p', class_='mathjax')
                    try:
                        abstract = abstract_element.text.strip()
                    # match any exception, consider specifying the exception types if known.
                    except Exception as e:
                        print(f"Error: {e}. Abstract not found, ending extraction.")
                        break
                    pdf_link_element = entry.find('a', title='Download PDF')
                    pdf_link = 'https://arxiv.org' + pdf_link_element['href'] if pdf_link_element else None
                    # translated_abstract = self.translate(abstract)
                    papers.append({'title': title,
                                   'authors': authors,
                                   'abstract': abstract,
                                   'link': pdf_link})
                                   #'translated_abstract': translated_abstract})

                return papers

            except (requests.exceptions.RequestException, Exception) as e:
                print(f"Error: {e}. Retrying {retries + 1}/{max_retries}...")
                retries += 1
                time.sleep(retry_delay)

        print(f"Failed to extract papers after {max_retries} retries.")
        return []  # Return an empty list if the maximum number of retries is reached






if __name__ == "__main__":
    classifier = LLMClassifier(model="gpt-3.5-turbo")  # Or set the OPENAI_API_KEY environment variable
    parser = PageParser()
    url = 'https://arxiv.org/list/cs.CL/new'
    papers = parser.extract_papers(url)
    num_of_papers = len(papers)
    print(f"Total papers: {num_of_papers}")

    llm_related_papers = []

    for i, paper in enumerate(papers):
        print(paper)
        title = paper['title']
        abstract = paper['abstract']
        print(f"({i+1}/{num_of_papers})Examining paper: {title}")
        is_llm_related, classification_answer = classifier.classify(title, abstract)
        if is_llm_related:
            llm_related_papers.append((paper, classification_answer))

    # file to save today's papers
    today = datetime.date.today()
    output_file = f'llm_related_papers_{today}.md'

    print(f"Total LLM related papers: {len(llm_related_papers)}")

    for i, (paper, classification_answer) in enumerate(llm_related_papers, 1):
        print(f"{i}. Title: {paper['title']}\n   Link: {paper['link']}\n   Classification Answer: {classification_answer}")
        # Save to file
        with open(output_file, 'a') as f:
            #f.write(f"## {i}. Title: {paper['title']}\n\n- **Authors**: {paper['authors']}\n- **Link**: {paper['link']}\n- **Classification Answer**: {classification_answer}\n\n- **Abstract**:\n\n{abstract}\n\n- **Translated Abstract**:\n\n{translated_abstract}\n\n---\n\n")
            # without translated abstract and classification answer
            f.write(f"## {i}. Title: {paper['title']}\n\n- **Authors**: {paper['authors']}\n- **Link**: {paper['link']}\n\n- **Abstract**:\n\n{abstract}\n\n---\n\n")