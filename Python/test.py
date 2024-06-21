def get_variable_name(var, scope):
    names = [name for name, value in scope.items() if value is var]
    return names

# Example usage
x = 10
y = 20
my_var = x

# Searching in globals
variable_names = get_variable_name(my_var, globals())
print("Variable names in globals:", variable_names)

# Example usage inside a function
def example_function():
    a = 30
    b = 40
    my_var_local = a

    # Searching in locals
    variable_names_local = get_variable_name(my_var_local, locals())
    print("Variable names in locals:", variable_names_local)

example_function()

