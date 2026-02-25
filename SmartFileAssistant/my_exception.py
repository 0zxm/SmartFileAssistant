

class NotSupportedSuffixException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        

    def __str__(self):
        return f"NotSupportedSuffixException: {self.message}"
    

class NotAllowedArgsException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        

    def __str__(self):
        return f"方法参数非法: {self.message}"