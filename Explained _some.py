# projecting Down to d Dimensions 
# rather than projecting them to the random 
# hyperplanes
# >selecting the best hyperplanes ensures that the projection will preserve as much varience  as possible 
# python code for the training set onto the plane defined by the first two princliple components
w2=Vt.t[:,:2]
x2d=x_centered.dot(w2)