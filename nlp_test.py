from nlp import Text
import string
from collections import Counter

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)


# initialize the text processor
text_processor = Text()

# load the stop words
stop_words = 'stop_words.txt'

# load pdf documents for processing
# change the file path where you have them saved bc it's prob different
screenplays = [
    ("a_real_pain_screenplay.pdf", "A Real Pain"),
    ("anatomy_of_a_fall_screenplay.pdf", "Anatomy of a Fall"),
    ("anora_screenplay.pdf", "Anora"),
    ("maestro_screenplay.pdf", "Maestro"),
    ("may_december_screenplays.pdf", "May December"),
    ("past_lives_screenplay.pdf", "Past Lives"),
    ("september_5_screenplay.pdf", "September 5"),
    ("the_brutalist_screenplay.pdf", "The Brutalist"),
    ("the_holdovers_screenplay.pdf", "The Holdovers"),
    ("the_substance_screenplay.pdf", "The Substance")
]

for filepath, label in screenplays:
    text_processor.load_text(filepath, label=label, stop_words=stop_words, parser=text_processor.pdf_parser)

# generate the sankey diagram for the top 5 words
text_processor.wordcount_sankey()

# display the second visualization
text_processor.frequency_heatmap()

# display the alternate second visualization
text_processor.frequency_barchart()

# display the complexity heatmap
text_processor.complexity_heatmap()


