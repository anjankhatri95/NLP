from kanren import Relation, facts, run, var
x=var() #declaring x variable with var
parent= Relation()
#presenting the fact in the system
facts(parent, ("Nathan","Mexican"),("Tina","Thai"),
              ("Tim", "Vegetarian"),("Sally","Italian"),
              ("John", "Thai"))

#setting the parameter and the output
Thai_food= run(2, x, parent(x, "Thai"))
print(Thai_food)

mexican_food= run(1, x, parent(x, "Mexican"))
print(mexican_food)

veg_food= run(1, x, parent(x, "Vegetarian"))
print(veg_food)

Italian_food= run(1, x, parent(x, "Italian"))
print(Italian_food)