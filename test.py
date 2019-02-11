import os 


from whoosh.fields import Schema, ID, KEYWORD, TEXT
from whoosh.index import create_in
from whoosh.query import Term

from pymongo import Connection
from bson.objectid import ObjectId

from whoosh.qparser import QueryParser
import PyPDF2

# Set index, we index title and content as texts and tags as keywords.
# We store inside index only titles and ids.
schema = Schema(content=TEXT(stored=True))

# Create index dir if it does not exists.
if not os.path.exists("index"):
    os.mkdir("index")

# Initialize index
index = create_in("index", schema)

def getPdf(file):
    data = []
    file = open(file, 'rb')
    pdfReader = PyPDF2.PdfFileReader(file)
    page_number = pdfReader.numPages
    for i in range(0, page_number):
        pageObj = pdfReader.getPage(i)
        data.append(pageObj.extractText())
    return '\n'.join(data)

text = getPdf("Data/1.pdf")
a = text.split('\n\n')

# Fill index with posts from DB
writer = index.writer()
for i in a:
    writer.add_document(content=i)

writer.commit()

gg = list()

with index.searcher() as searcher:
    query = QueryParser("content", index.schema).parse(u'Holding Pattern')
    results = searcher.search(query)
    for result in results:
        print(result)
        gg.append(dict(result))