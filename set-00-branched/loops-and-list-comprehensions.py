# Databricks notebook source
# MAGIC %md
# MAGIC # Loops
# MAGIC 
# MAGIC Loops are a way to repeatedly execute some code. Here's an example:

# COMMAND ----------

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line

# COMMAND ----------

# MAGIC %md
# MAGIC The ``for`` loop specifies 
# MAGIC - the variable name to use (in this case, `planet`)
# MAGIC - the set of values to loop over (in this case, `planets`)
# MAGIC 
# MAGIC You use the word "``in``" to link them together.
# MAGIC 
# MAGIC The object to the right of the "``in``" can be any object that supports iteration. Basically, if it can be thought of as a group of things, you can probably loop over it. In addition to lists, we can iterate over the elements of a tuple:

# COMMAND ----------

multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product

# COMMAND ----------

# MAGIC %md
# MAGIC You can even loop through each character in a string:

# COMMAND ----------

s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')        

# COMMAND ----------

# MAGIC %md
# MAGIC ### range()
# MAGIC 
# MAGIC `range()` is a function that returns a sequence of numbers. It turns out to be very useful for writing loops.
# MAGIC 
# MAGIC For example, if we want to repeat some action 5 times:

# COMMAND ----------

for i in range(5):
    print("Doing important work. i =", i)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ``while`` loops
# MAGIC The other type of loop in Python is a ``while`` loop, which iterates until some condition is met:

# COMMAND ----------

i = 0
while i < 10:
    print(i, end=' ')
    i += 1 # increase the value of i by 1

# COMMAND ----------

# MAGIC %md
# MAGIC The argument of the ``while`` loop is evaluated as a boolean statement, and the loop is executed until the statement evaluates to False.

# COMMAND ----------

# MAGIC %md
# MAGIC # List comprehensions
# MAGIC 
# MAGIC List comprehensions are one of Python's most beloved and unique features. The easiest way to understand them is probably to just look at a few examples:

# COMMAND ----------

squares = [n**2 for n in range(10)]
squares

# COMMAND ----------

# MAGIC %md
# MAGIC Here's how we would do the same thing without a list comprehension:

# COMMAND ----------

squares = []
for n in range(10):
    squares.append(n**2)
squares

# COMMAND ----------

# MAGIC %md
# MAGIC We can also add an `if` condition:

# COMMAND ----------

short_planets = [planet for planet in planets if len(planet) < 6]
short_planets

# COMMAND ----------

# MAGIC %md
# MAGIC (If you're familiar with SQL, you might think of this as being like a "WHERE" clause)
# MAGIC 
# MAGIC Here's an example of filtering with an `if` condition *and* applying some transformation to the loop variable:

# COMMAND ----------

# str.upper() returns an all-caps version of a string
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
loud_short_planets

# COMMAND ----------

# MAGIC %md
# MAGIC People usually write these on a single line, but you might find the structure clearer when it's split up over 3 lines:

# COMMAND ----------

[
    planet.upper() + '!' 
    for planet in planets 
    if len(planet) < 6
]

# COMMAND ----------

# MAGIC %md
# MAGIC (Continuing the SQL analogy, you could think of these three lines as SELECT, FROM, and WHERE)
# MAGIC 
# MAGIC The expression on the left doesn't technically have to involve the loop variable (though it'd be pretty unusual for it not to). What do you think the expression below will evaluate to? Press the 'output' button to check.

# COMMAND ----------

[32 for planet in planets]

# COMMAND ----------

# MAGIC %md
# MAGIC List comprehensions combined with functions like `min`, `max`, and `sum` can lead to impressive one-line solutions for problems that would otherwise require several lines of code. 
# MAGIC 
# MAGIC For example, compare the following two cells of code that do the same thing.

# COMMAND ----------

def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative

# COMMAND ----------

# MAGIC %md
# MAGIC Here's a solution using a list comprehension:

# COMMAND ----------

def count_negatives(nums):
    return len([num for num in nums if num < 0])

# COMMAND ----------

# MAGIC %md
# MAGIC Much better, right?
# MAGIC 
# MAGIC Well if all we care about is minimizing the length of our code, this third solution is better still!

# COMMAND ----------

def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])

# COMMAND ----------

# MAGIC %md
# MAGIC Which of these solutions is the "best" is entirely subjective. Solving a problem with less code is always nice, but it's worth keeping in mind the following lines from [The Zen of Python](https://en.wikipedia.org/wiki/Zen_of_Python):
# MAGIC 
# MAGIC > Readability counts.  
# MAGIC > Explicit is better than implicit.
# MAGIC 
# MAGIC So, use these tools to make compact readable programs. But when you have to choose, favor code that is easy for others to understand.

# COMMAND ----------

# MAGIC %md
# MAGIC # Your Turn
# MAGIC 
# MAGIC You know what's next -- we have some **[fun coding challenges](https://www.kaggle.com/kernels/fork/1275177)** for you! This next set of coding problems is shorter, so try it now.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/python/discussion) to chat with other learners.*
