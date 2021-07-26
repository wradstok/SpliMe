from typing import Dict

# Maps
Time = int
Id = int
Name = str

Name2Id = Dict[Name, Id]
Id2Name = Dict[Id, Name]
Time2Id = Dict[Time, Id]
Time2Name = Dict[Time, Name]

IdMap = Dict[Id, Time2Id]
