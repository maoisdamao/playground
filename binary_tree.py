class Node:
    def __init__(self, data=None):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        if self.data is not None:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

    def lookup(self, data, parent=None):
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        return data, parent

    def children_count(self):
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt

    def delete(self, data):
        node, parent = self.lookup(data)
        if node is not None:
            children_count = node.children_count()
            if children_count == 0:
                if parent:
                    if parent.left is node:
                        parent.left = None
                    else:
                        parent.right = None
                    del node
                else:
                    self.data = None
            elif children_count == 1:
                n = node.left or node.right
                if parent:
                    if parent.left is node:
                        parent.left = n
                    else:
                        parent.right = n
                    del node
                else:
                    self.left = n.left
                    self.right = n.right
                    self.data = n.data
            else:
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                node.data = successor.data
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right

    def bfs(self):
        q = []
        output_tree = []
        if self.data is None:
            return
        q.append(self)
        while q:
            node = q.pop(0)
            output_tree.append(node.data)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        print("breadth first traversal:"+"".join(output_tree))

    def dfs(self):
        pass


if __name__ == "__main__":
    root = Node()
    elements = "abcdefg"
    for item in elements:
        root.insert(item)
    root.bfs()  # abcdefg
