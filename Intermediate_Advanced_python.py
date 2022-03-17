# python @classmethod and @staticmethod
# used when you don't need the object

class Person:
    population = 50
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age
        
    @classmethod
    def getPopulation(cls):
        return cls.population
    
    @staticmethod
    def isAdult(age):
        return age >= 18
    
newPerson = Person("Alex", 26)

print(Person.getPopulation())
print(Person.isAdult(20))