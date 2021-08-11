class calculator():
    def __init__(self):

        # Getting the value from user

        value=input('enter input:')
        
        # Sorting the value for different operation

        if '+' in value:
            # splitting on the basis of operator we are getting

            # 0 represent first index
            self.a = value.split('+')[0]

            # -1 represent last index
            self.b = value.split('+')[-1]
            # printing the returned value on returned number and calling the function in the print function
            print(self.add())

        elif '-' in value:
            self.a = value.split('-')[0]
            self.b = value.split('-')[-1]
            print(self.sub())

        elif '*' in value:
            self.a = value.split('*')[0]
            self.b = value.split('*')[-1]
            print(self.mul())

        elif '/' in value and value.count('/')==1:
            self.a = value.split('/')[0]
            self.b = value.split('/')[-1]
            print(self.div())

        elif '//' in value and value.count('/')==2:
            self.a = value.split('//')[0]
            self.b = value.split('//')[-1]
            print(self.divRound())

        elif '%' in value:
            self.a = value.split('%')[0]
            self.b = value.split('%')[-1]
            print(self.mod())
            
        else:
            print("Invalid Input")

# function for operation like +,-,*,/,//,%


    def add(self):
        # when the numbers are getting stored it will be in string so coverting it into number format using eval() function
        return eval(self.a)+eval(self.b)

    def sub(self):
        return eval(self.a)-eval(self.b)
    
    def div(self):
        return eval(self.a)/eval(self.b)
    
    def divRound(self):
        return eval(self.a)//eval(self.b)
    
    def mul(self):
        return eval(self.a)*eval(self.b)
    
    def mod(self):
        return eval(self.a)%eval(self.b)

# Creating the object to call the init function

abc=calculator()