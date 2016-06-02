import psycopg2
from ddlite_parser import *

def split_arrays_if_necessary(sentence):
    for property_ in ['words', 'lemmas', 'poses', 'dep_parents',
                      'dep_labels', 'token_idxs']:
        property_value = getattr(sentence, property_, "")
        if type(property_value) is str:
            sentence = sentence._replace(**{property_: property_value.split("|^|")})
    return sentence

def sentences_from_db(attribute_list, dbname, table="sentences_input", dbhost="localhost",
                      dbuser="postgres", dbpass=None, dbport=5432, count=None):
    """
    Create a list of sentences from stuff
    """
    conn = psycopg2.connect(database=dbname, user=dbuser, password=dbpass,
                            host=dbhost, port=dbport)

    cur = conn.cursor()
    
    query = "SELECT {} FROM {}".format(", ".join(attribute_list), "sentences_input")
    if count is not None:
        query = query + " ORDER BY random() LIMIT {}".format(count)

    cur.execute(query)
    return [split_arrays_if_necessary(Sentence(*row)) for row in cur]

print list(sentences_from_db(['words', 'lemmas', 'poses', 'dep_parents', 'dep_paths',
                             'sent_id', 'doc_id', "replace(words, '|^|', ' ')", '0', 'section_id'],
                            dbname='genomics_full',
                            dbuser='byoo1',
                            dbhost='raiders7.stanford.edu',
                            dbport='6432',
                            count=10))
