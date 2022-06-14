class GraphNode():
    def __init__(self, name, id, type, level = None):
        assert type == "category" or type == "page"
        self.name = name
        self.id = id
        self.type = type
        self.level = level
        self.parents = []
        self.subcategories = []
        self.pages = []
        self.siblings = []
    
    def add_child(self, name, type):
        assert type == "category" or type == "page"
        if type == "category":
            if not name in self.subcategories:
                self.subcategories.append(name)
        else:
            if not name in self.pages:
                self.pages.append(name)
        
    def add_parent(self, name):
        if not name in self.parents:
            self.parents.append(name)
        
    def add_sibling(self, name):
        if not name in self.siblings:
            self.siblings.append(name)

 

    


