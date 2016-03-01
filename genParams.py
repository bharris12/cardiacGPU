import random
length_of_cables = input("Enter the number of cells in a cable you want: ")

number_of_cables = input("Enter the number of cables you want: ")

#num_params = input("Enter the number of paramaters you want to randomize: ")
num_params = 17
#for i in range(0, num_params-1):
#    param_value[i] = input("enter the #%d parameter that you want to change" % (i + 1,))
param_value = [14.838, 5.405, .153, .0146, .1238, 2.9e-4, 5.92e-4, .073, .392, 3.98e-5, 2.724, 1000, 3.6e-4, 6.375, .102, 1.27, 526]
print(len(param_value))
file_name = raw_input("Create a name for the parameter set you want to create: ")

file_name_1D = file_name + "_1D.txt"
file_name_2D = file_name + "_2D.txt" 
print("Building text files named " + file_name_1D + " and " + file_name_2D)

file_1D = open(file_name_1D, 'w')
file_2D = open(file_name_2D, 'w')

for i in range(1,number_of_cables):
    
    for j in range(1,num_params):
        rand_param_1 = random.gauss(0,.15)
        file_1D.write(str(param_value[i]*rand_param_1)+ '\t') # writes value for the 1 cell in a cable
        file_2D.write(str(param_value[i]*rand_param_1)+ '\t') # same first value in 1D and 2D cables
        for n in range(2,length_of_cables):
            rand_param_n = random.gauss(rand_param_1, .15)  # generates value normally distributed around the first parameter
            file_1D.write(str(rand_param_1 * param_value[i]) + '\t') # repeats first value for 1D
            file_2D.write(str(rand_param_n * param_value[i]) + '\t') # enters new value for 2D
    file_1D.write("\n")
    file_2D.write("\n")    
    
file_1D.close()
file_2D.close()
print("done")
    