import ConfigParser
import psycopg2
import psycopg2.extensions
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)

from ddlite_parser import *

def split_arrays_if_necessary(sentence):
    for property_ in ['words', 'lemmas', 'poses', 'dep_parents',
                      'dep_labels', 'token_idxs']:
        property_value = getattr(sentence, property_, "")
        if isinstance(property_value, basestring):
            sentence = sentence._replace(**{property_: property_value.split(u'|^|')})
        elif type(property_value) is int:
            sentence = sentence._replace(**{property_: [property_value]})
        elif type(property_value) is not list:
            sentences = sentence._replace(**{property_: list(property_value)})
    return sentence

class DatabaseHandler():
	dbname = dbuser = dbpass = dbhost = dbport = ''	
	def __init__(self, inifile):
		config = ConfigParser.ConfigParser(allow_no_value=True)
		config.read(inifile)
		assert 'Connection' in config.sections() and len(config.sections()) == 1
		self.dbname = config.get('Connection','dbname')
		self.dbuser = config.get('Connection','dbuser')
		self.dbpass = config.get('Connection','dbpass')
		self.dbhost = config.get('Connection','dbhost')
		self.dbport = config.get('Connection','dbport')

	def get_random_sentences(self,attribute_list=['words', 'lemmas', 'poses', 'dep_parents', 'dep_paths','sent_id', 'doc_id', 'replace(words, \'|^|\', \' \')', '0', 'section_id'], table="sentences_input", count=100):
	    conn = psycopg2.connect(database=self.dbname, user=self.dbuser, password=self.dbpass,
				    host=self.dbhost, port=self.dbport)

	    cur = conn.cursor()
	    
	    query = "SELECT {} FROM {} s, (SELECT doc_id as document_id FROM document_ids ORDER BY RANDOM() LIMIT {}) d where d.document_id = s.doc_id".format(", ".join(attribute_list), table, count)
	    cur.execute(query)
	    return [split_arrays_if_necessary(Sentence(*row)) for row in cur]

