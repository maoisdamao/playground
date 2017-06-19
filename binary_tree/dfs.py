def preorder_visit(node, output=[]):
    # node is an object of Node
    output.append(str(node.data))
    # print(output)
    if node.left:
        preorder_visit(node.left, output)
    if node.right:
        preorder_visit(node.right, output)


def inorder_visit(node, output=[]):
    # node is an object of Node
    if node.left:
        inorder_visit(node.left, output)
    output.append(str(node.data))
    if node.right:
        inorder_visit(node.right, output)


def postorder_visit(node, output=[]):
    # node is an object of Node
    if node.left:
        postorder_visit(node.left, output)
    if node.right:
        postorder_visit(node.right, output)
    output.append(str(node.data))
