import os
import cPickle as pickle
import itertools

from ete3 import Tree

from .constants import *
from .util import *

EXCLUDE = {'sp0c326f', 'sp097c54'}

class NTBDB(object):
    def __init__(self):
        with open(os.path.join(BASE_DIR, 'metadata.pickle')) as md:
            self.metadata = pickle.load(md)
        for img in EXCLUDE:
            del self.metadata[img]
        self.by_tag = dict()
        for p in self.metadata.itervalues():
            for tag in p['tags']:
                self.by_tag.setdefault(tag, []).append(p)

        self.tags = Tree(os.path.join(BASE_DIR, 'tags.nw'), format=8)
        self.tag_by_name = {tag.name: tag for tag in self.tags.get_descendants()}
     
    def by_tag_with_children(self, tag_name):
        tag_node = self.tags.search_nodes(name=tag_name)[0]
        all_tags = [tag_node]
        all_tags.extend(tag_node.get_descendants())
        return list(itertools.chain.from_iterable(self.by_tag.get(tag.name, []) for tag in all_tags))
    
    def tag_score(self, tag):
        return len(self.by_tag.get(tag.name, [])) + sum(map(self.tag_score, tag.children))
    
    def top_tags(self, max_children=5, max_depth=2):
        top_tags = self.tags.copy()
        for n in top_tags.traverse():
            n.children = sorted(n.children, key=self.tag_score, reverse=True)[:max_children]
            if n.get_distance(n.get_tree_root()) > max_depth - 1:
                n.children = []
        return top_tags