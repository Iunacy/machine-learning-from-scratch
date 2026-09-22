
target = 6.0
feature = 0.5

weight = 6.0
bias = 1.0
learning_rate = 0.5

for step in range(5):
   new_prediction=feature*weight+bias
   new_error=new_prediction-target
   new_loss=new_error**2

   new_weight_gradient=2*new_error*feature

   new_bias_gradient=2*new_error

   print(step, new_prediction, new_loss, weight, bias)

   new_weight=weight-learning_rate*new_weight_gradient
   weight=new_weight

   

   new_bias=bias-learning_rate*new_bias_gradient
   bias=new_bias
   