"""
File: nlp.py

Description: A resuable, extensible framework for compartive text analysis
            designed to work with any arbitrary collection of related docs
"""

from collections import Counter, defaultdict
import string
# pip install pdfplumber
# pip install textstat
import pdfplumber
import plotly.graph_objects as go
import numpy as np
import textstat
from plotly.subplots import make_subplots

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

class Text:

    def __init__(self):
        """ Constructor """
        self.data = defaultdict(dict)

    def load_stop_words(self, stopfile):
        """ Load a list of common or stop words.
        These get filtered from each file automatically"""
        with open(stopfile, 'r') as file:
            stop_words = [line.strip() for line in file if line.strip()]

        return stop_words

    def pdf_parser(self, filename, stopfile):
        """ For processing pdf documents using pdfplumber """
        # load stop words
        stop_words = set(self.load_stop_words(stopfile))

        # initialize raw text storage
        raw = ""

        # use pdfplumber to open and process the pdf file
        with pdfplumber.open(filename) as pdf_file:
            num_pages = len(pdf_file.pages)

            # iterate over each page in the pdf
            for page in pdf_file.pages:
                text = page.extract_text()
                if text:  # check if text was extracted
                    raw += text + "\n"  # append text from each page

        # convert to lowercase, remove punctuation
        raw = raw.lower()
        raw = raw.translate(str.maketrans('', '', string.punctuation))

        # use regex to only keep alphabetic characters (ignore "-" and others)
        words = [word for word in raw.split() if word.isalpha() and word not in stop_words]

        # count word occurrences
        wc = Counter(words)
        num_words = len(words)

        # return results as a dictionary
        results = {'wordcount': wc, 'num_words': num_words, 'num_pages': num_pages, 'raw_text': raw}

        return results

    def simple_text_parser(self, filename, stop_words):
        """ For processing simple, unformatted text documents """
        with open(filename, 'r') as file:
            raw = file.read().lower()

        # remove punctuation
        raw = raw.translate(str.maketrans('', '', string.punctuation))

        # use regex to only keep alphabetic characters (ignore "-" and others)
        words = [word for word in raw.split() if word.isalpha() and word not in stop_words]

        # get word count
        wc = Counter(words)

        return {'wordcount': wc, 'num_words': len(words)}

    def load_text(self, filename, label=None, stop_words=None, parser=None):
        """ Register a text file with the library.
        Label is an optional label to use in visualizations to identify the text """

        if stop_words is None:
            stop_words = self.load_stop_words('stop_words.txt')

        if parser is None:
            results = self.simple_text_parser(filename)
        else:
            results = parser(filename, stop_words)

        if label is None:
            label = filename

        # Store results in the data dictionary (self.data)
        for k, v in results.items():
            self.data[k][label] = v


    def wordcount_sankey(self, word_list=None, k=5):
        """ Map each text to words using a Sankey diagram."""
        # if no word_list is given, use k_most frequent words from each document
        files = list(self.data['wordcount'].keys())

        if word_list is None:
            top_words_per_file = {}
            for label in files:
                wordcount = self.data["wordcount"][label]
                top_words = [word for word, _ in wordcount.most_common(k)]
                top_words_per_file[label] = top_words

        else:
            # if a word_list is given, use it
            top_words_per_file = {label: word_list for label in files}

        # prepare for sankey
        sources = []
        targets = []
        values = []

        for label, wordcount in self.data["wordcount"].items():
            top_words = top_words_per_file[label]

            for word in top_words:
                if word in wordcount:
                    sources.append(label)
                    targets.append(word)
                    values.append(wordcount[word])

        # unique ids for sources (files) and targets (words)
        all_nodes = list(set(sources + targets))
        node_map = {node: i for i, node in enumerate(all_nodes)}  # mapping

        # map sources and targets to node indices
        sources_indices = [node_map[source] for source in sources]
        targets_indices = [node_map[target] for target in targets]

        # define a list of colors for the Sankey diagram
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63A1', '#FF77AC', '#119DFF', '#F7B801', '#00FF00', '#FF4500']
        node_colors = {node: colors[i % len(colors)] for i, node in enumerate(all_nodes)}  # cycle through colors

        # assign colors for links to sources
        link_colors = [node_colors[source] for source in sources]

        # create the sankey diagram
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,  # labels for nodes
                color=[node_colors[node] for node in all_nodes]  # apply color to nodes
            ),
            link=dict(
                source=sources_indices,
                target=targets_indices,
                value=values,  # flow size aka word count
                color=link_colors  # apply color to links
            )
        ))

        fig.update_layout(
            title_text="Wordcount Sankey",
            font_size=10,
            height=600,
            width=900
        )
        fig.show()

    def frequency_heatmap(self, k=10):
        """ A visualization array of subplots with one subplot for each text file"""
        files = list(self.data['wordcount'].keys())

        # get k most common words across all documents
        word_freq = Counter()
        for label in files:
            word_freq.update(self.data['wordcount'][label])

        top_words = [word for word, _ in word_freq.most_common(k)]

        # make frequency matrix
        freq_matrix = np.array([[self.data['wordcount'][file].get(word, 0) for file in files] for word in top_words])

        # create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=freq_matrix,
            x=files,
            y=top_words,
            colorscale='Blues'
        ))

        fig.update_layout(
            title="Word Frequency Heatmap",
            xaxis_title="Text Files",
            yaxis_title="Words",
            height=500,
            width=1000
        )
        fig.show()

    def frequency_barchart(self, k=10):
        """ Creates a grid of bar charts (one per screenplay) of top k words"""

        files = list(self.data['wordcount'].keys())
        num_files = len(files)

        # determine subplot grid layout
        cols = 2
        rows = (num_files + 1) // cols

        # initialize fig
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=files)

        # iterate through files, making a bar chart for each
        for idx, label in enumerate(files):
            row = idx // cols + 1
            col = idx % cols + 1

            wordcount = self.data['wordcount'][label]
            top_words = wordcount.most_common(k)

            words = [word for word, _ in top_words]
            counts = [count for _, count in top_words]

            # add bar chart to fig
            fig.add_trace(
                go.Bar(x=words, y=counts, name=label),
                row=row, col=col
            )

        fig.update_layout(
            title = 'Top-k words per screenplay',
            height = 300 * rows,
            width = 900,
            showlegend = False
        )

        fig.show()

    def get_flesch_kindcaid_grade(self, text):
        """ calculate the flesch-kincaid grade level of a text """
        return textstat.flesch_kincaid_grade(text)

    def get_polysyllable_count(self, text):
        """ calculate the number of polysyllables in a text """
        return textstat.polysyllabcount(text)

    def get_features(self, text):
        """ gets the features of the text for the heatmap visualization """
        word_count = self.data['wordcount'].get(text)
        grade_level = self.get_flesch_kindcaid_grade(text)
        polysyllable_count = self.get_polysyllable_count(text)

        return {
            'Word Count': word_count,
            'Grade Level': grade_level,
            'Polysyllable Count': polysyllable_count
        }

    def complexity_heatmap(self, k=10):
        """ visualizes the word count, polysyllable count, and grade level of a text """

        # prepare data for heatmap
        data = []
        labels = []

        for label in list(self.data['wordcount'].keys())[:k]:
            text = ' '.join(self.data['wordcount'][label].keys())

            num_words = self.data['num_words'].get(label)
            grade = self.get_flesch_kindcaid_grade(text)
            polysyllable_count = self.get_polysyllable_count(text)

            data.append([num_words, polysyllable_count, grade])
            labels.append(label)

            # normalize values
            max_word_count = max([item[0] for item in data])
            max_polysyllable_count = max([item[1] for item in data])
            max_grade = max([item[2] for item in data])

            normalized_data = [
                [num_words / max_word_count, polysyllable_count / max_polysyllable_count, grade / max_grade]
                for num_words, polysyllable_count, grade in data
            ]

        normalized_data = np.array(normalized_data)

        # create heatmap

        fig = make_subplots(
            rows=1, cols=1,
        )

        heatmap = go.Heatmap(
            z=normalized_data.T,
            x=labels,
            y=['Word Count', 'Polysyllable Count', 'Grade Level'],
            colorscale='algae',
            colorbar=dict(title="Normalized Text Feature Values"),
            showscale=True
        )

        fig.add_trace(heatmap)

        fig.update_layout(
            title = "Screenplay Complexity Heatmap",
            xaxis_title="Screenplays",
            yaxis_title="Complexity",
            height=600,
            width=800,
        )

        fig.show()