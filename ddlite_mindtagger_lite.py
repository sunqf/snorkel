import numpy as np

class TaggerRow:
  def __init__(self, T, c_id):
    self._T = T
    self._j = c_id
  
  def get_row(self, attrib, tag):
    # Set new tag
    self._T._tags[self._j] = tag
    # Get content
    content = self._T._entity_cand_to_html(self._j, attrib)
    t = self._T._types[self._j]
    lfs = self._T._lf_indicators[self._j]
    return "<td>{}</td> <td>{}</td> <td>{}</td>".format(content, t, lfs)

class TaggerTable:
  def __init__(self, cands, current_tags, cand_types, lf_matrix_sub, lf_nm):
    self._C = cands
    self._tags = current_tags
    self._types = cand_types
    self._lf = lf_matrix_sub
    self._lf_nm = lf_nm
    self._neg_color = "<font color=\"#F87217\">"
    self._pos_color = "<font color=\"#169DF7\">"
    self._ec = "</font>"
    
  def _get_attrib_seq(self, j, attrib):
    """ Helper util to get the match attrib of the input context """
    try:
      return self._C[j][attrib]
    except TypeError:
      return self._C[j].__dict__[attrib]

  def _entity_cand_to_html(self, j, attrib):
    """ Return the joined candidate sequence with highlighted candidate """
    seq = self._get_attrib_seq(j, attrib)
    highlighted = ["<mark>{}</mark>".format(seq[i]) if i in c.idxs else seq[i]
                   for i in xrange(len(seq))]
    return " ".join(highlighted)

  def _lf_indicators(self, j):
    """ Return the color coded LFs that labeled the candidate """
    c_labels = np.ravel(self._lf[j, :])
    labeled = np.where(c_labels != 0)
    labeled_nm = [self._lf_nm[i] for i in labeled]
    labeled_l = c_labels[labeled]
    return "\n".join(["{}{}{}".format(self._pos_color, nm, self._ec)
                      if lb > 0 else
                      "{}{}{}".format(self._neg_color, nm, self._ec)
                      for nm,lb in zip(labeled_nm, labeled_l)]):

  def 
  
  
  