class Stack:
    data = list()
    def __init__(self):
        pass

    def push(self, value):
        self.data.append(value)

    def pop(self):
        return self.data.pop()

    def peek(self):
        return self.data[len(self.data) - 1]
