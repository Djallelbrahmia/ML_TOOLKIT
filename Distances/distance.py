from abc import ABC, abstractmethod

class Distance(ABC):
    @abstractmethod
    def calculate(self, point1, point2):
        pass