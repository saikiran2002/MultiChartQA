import openai
import pandas as pd
import base64

class ChartAnalysisWorkflow:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.df = pd.read_excel("ChartswithCaptions.xlsx")
        self.chart_analysis_prompt = """
            You are GPT-4o-mini, a reasoning model specialized in analyzing charts and their captions. Your task is to carefully examine the provided chart and caption, then clearly identify and list the attributes used in the chart.

            When presented with a chart and its caption:
                1. Carefully inspect the chart provided.
                2. **Let's think step-by-step.** First, carefully inspect the provided chart and caption. Identify and list all attributes used in the chart. 
                3. **Briefly explain** how each attribute is represented or visualized in the chart (e.g., axis labels, legends, colors, data points).
                4. **Clearly separate** your response into two sections:
                    a. "**Identified Attributes:**" (list attributes succinctly)
                    b. "**Visualization Explanation:**" (briefly describe how each attribute is visualized)
                5. **Clearly summarize** the specific chart variables used (e.g., x-axis, y-axis, legend, color encoding) separately at the end of your response.

            Important instructions for optimal performance:

            1. Be concise and direct in your analysis.
            2. Do not include unnecessary examples or additional context beyond what is provided.
            3. Do not generate or assume any information not explicitly present in the provided chart and caption.
            4. Leverage your internal chain-of-thought reasoning capability without explicit prompting for step-by-step reasoning.
            5. Ensure your output is accurate, consistent, and directly based on the provided data only.
            """
        self.keyword_extraction_prompt = """ 
            You are a specialized Keyword Extraction GPT designed to distill chart analyses into semantic keywords.

            Your task is to analyze the provided chart description (which includes "Identified Attributes", "Visualization Explanation", and "Summary of Chart Variables") and extract the most significant keywords that represent:

            1. Core variables/attributes present in the chart
            2. Key measurement units and scales
            3. Central scientific or data concepts being visualized
            4. Important methodologies or data types

            Guidelines:
            - Focus on domain-specific terminology rather than generic terms
            - Include units of measurement when semantically relevant
            - Extract only terms explicitly present or strongly implied in the analysis
            - Limit to 8-12 keywords for clarity

            Structure your response exactly as follows:

            ## Extracted Keywords:
            - [Keyword 1]
            - [Keyword 2]
            ...

            Do not include any explanations or additional commentary.
            """
        self.comparison_prompt = """
            You are a Comparison Agent specialized in identifying relationships between different charts based on their attributes and extracted keywords.

            Given two chart analyses with their extracted keywords:

            Chart A Keywords:
            [Keywords from Chart A]

            Chart B Keywords:
            [Keywords from Chart B]

            Perform the following analysis:

            1. Identify common attributes shared by both charts
            2. Identify unique attributes specific to each chart
            3. Describe potential meaningful relationships between these charts

            Structure your response exactly as follows:

            ## Common Attributes:
            - [List common attributes]

            ## Unique Attributes - Chart A:
            - [List attributes unique to Chart A]

            ## Unique Attributes - Chart B:
            - [List attributes unique to Chart B]

            ## Potential Relationships:
            - [Describe potential relationships between charts]

            Be concise, specific, and focus only on meaningful connections.
            """
        self.bridging_prompt = """
            You are an expert Data Analysis GPT specialized in generating comprehensive analytical insights from multiple data visualizations.

            Given two chart analyses (including their identified attributes, visualization explanations, and variables), your task is to:

            1. Generate thoughtful questions that can ONLY be answered by analyzing both charts together
            2. Provide detailed answers to each question based on the chart analyses
            3. Explain why each question matters from an analytical perspective

            Your questions should:
            - Require synthesizing information across both visualizations
            - Reveal meaningful relationships, contrasts, or complementary information
            - Consider temporal, spatial, or conceptual connections between the visualizations
            - Lead to deeper insights about patterns, relationships, or contradictions

            Structure your response exactly as follows:

            ## Analytical Questions and Answers:

            ### Question 1:
            [Question that requires analyzing both charts together]

            #### Answer:
            [Detailed answer based on analysis of both charts]

            #### Why This Question Matters:
            [Explanation of the analytical significance and what insights this reveals]

            ### Question 2:
            [Question that requires analyzing both charts together]

            #### Answer:
            [Detailed answer based on analysis of both charts]

            #### Why This Question Matters:
            [Explanation of the analytical significance and what insights this reveals]

            [Continue with additional questions as appropriate]

            Ensure each question is specific, data-focused, and genuinely requires examining both charts to answer effectively. Your answers should demonstrate cross-chart analysis and synthesis of information.

            """
    
    def encode_image(self,image_path):

        caption = self.df[self.df["imageid"]==int(image_path.split("/")[1].split(".")[0])]["full_caption"].values[0]

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8"), caption

    def analyze_chart(self, image_path):

        chart_image, chart_caption = self.encode_image(image_path)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.chart_analysis_prompt},
                {"role": "user",
                    "content": [
                        {"type": "text", "text": f"Here is the caption for the image: {chart_caption}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{chart_image}"
                            }
                        }
                    ],
                }
            ]
        )
        return response.choices[0].message.content
    
    def extract_keywords(self, chart_analysis):

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.keyword_extraction_prompt},
                {"role": "user", "content": [
                {"type": "text", "text": f"Here is the Chart Analysis output for chart: {chart_analysis}"}
            ],}
            ]
        )
        return response.choices[0].message.content
    
    def compare_charts(self, keywords_a, keywords_b):

        prompt = f"Chart A Keywords:\n{keywords_a}\n\nChart B Keywords:\n{keywords_b}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.comparison_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def generate_questions(self, analysis_a, analysis_b, comparison):

        prompt = f"Chart A Analysis:\n{analysis_a}\n\nChart B Analysis:\n{analysis_b}\n\nComparison:\n{comparison}"
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.bridging_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def run_complete_workflow(self, chart_a_image, chart_b_image):

        analysis_a = self.analyze_chart(chart_a_image)
        analysis_b = self.analyze_chart(chart_b_image)
        
        keywords_a = self.extract_keywords(analysis_a)
        keywords_b = self.extract_keywords(analysis_b)
        

        comparison = self.compare_charts(keywords_a, keywords_b)
        
        questions = self.generate_questions(analysis_a, analysis_b, comparison)
        
        return {
            "chart_a_analysis": analysis_a,
            "chart_b_analysis": analysis_b,
            "keywords_a": keywords_a,
            "keywords_b": keywords_b,
            "comparison": comparison,
            "analytical_questions": questions
        }


        