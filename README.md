# Sentiment-Analysis
Analyzing sentiment from a collection of journal (or any other) entries.

## Structure of the journal
- The journal entries are made in the `data` folder, in the file named `test_journal.txt`
- The delimiter between entries is "---\n" i,e; 3 hyphens followed by a newline.
- The first piece of information in the entry in the date (yyyy/mm/dd) followed by a newline ("\n").
- Then comes the content of the journal.
- Then comes the delimiter before the next entry, and so on.

## Usage
- Do `pip install -r requirements.txt` to install the dependencies inside your virtual environment.
- Make entries into `test_journal.txt`
- Navigate into the `script` folder.
- Do `python3 code.py` or `python code.py` or `py code.py`
- Head over to the `results` folder to view your results.
  - There will be 3 PNG files.
  - Polarity score between -1 and +1 that tells how negative or positive the journal's content is.
  - Subjectivity score between 0 and +1 that tells how subjective (closer to 0) or objective (closer to 1) the journal's content is.
  - WordCloud of the most frequently used words in the journal.
