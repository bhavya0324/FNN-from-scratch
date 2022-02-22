<strong>Dataset</strong>. : FASHION MNIST <br />

<strong>About Dataset</strong> : we have 60000 datapoints as training images and 10000 datapoints as test images with each image is a standardized 28*28 size in gray scale (i.e., 784 pixels) <br/>

<strong> Optimizers for Back_Propagation used </strong> : stochastic gradient descent, momemtum based gradient descent,nestrov gradient descent,rmsprop, adam, nadam <br/>

<strong> Training Model and Hyperparameter Tuning </strong> : <br/>

We have split the training data into 90:10 ratio in a random fashion using stratify that is provided by sklearn library as follows :</br>
```
train_x, valid_x, cat_train_y, cat_val_y = train_test_split(flatted_train_images, train_labels, test_size=0.1, stratify = train_labels ,random_state=42)
```

i.e., 54000 data points belong to training data and 6000 data points belong to validation data </br>

Optimizers mentioned above are used to learn parameters for our neural network which are Weights(W) and Biases(b) using back propagation and sweep configuration offered by wandb is used to tune hyperparameters.

The following hyperparameters: <br/>
epochs , hl (number of hidden layers) , hs (size of each hidden layer) , learning_rate , optimizer , batch_size , initialization_strategy , activation_function are tuned using wandb as follows : <br/>

If you want to add some new parameter change the sweep configuration in below code i.e., add parameters in the below sample configuration : <br/>
```
sweep_config_temp={
  "parameters": {
        "hidden_layers":{
            "values":[3,4,5]
        },
        "optimizer":{
          "values":["adam","nadam"]
        }
    }
}
```

After setting up the sweep config and invoking it using the command ```wandb.agent(sweep_id,train)``` ,
train function will be invoked for every combination possible for the hyperparameter choices given in the sweep configuration and corresponding optimizer that matched with any in ```"optimizer":{"values":['nadam']}``` will be called further. <br/>

<strong>Evaluating a model </strong> : <br/>

We are evaluating the model using mainly ```cross_entropy_loss``` but if you want to evaluate the model using squared loss then this choice should be added in the train_function as follows : <br/>
```
def train():
  run=wandb.init()
  config = wandb.config
 
  loss_fun='squared_loss'
```
<strong> Flexibility of the code </strong> : <br/>

If you want to add new Optimizer for example 'eve' do the following steps : <br/>

<strong> step1</strong> : <br/> Write the eve optimizer code in a function and name it as eve and then Add eve as optimizer value in the sweep configuration as follows : <br/>
```
"optimizer":{ "values" : ['eve'] }
```
<strong>step2</strong> : <br/> Add the following two lines of code in train function : <br/>
```
if config.optimizer == 'eve':
    val_acc = eve(train_x,train_y,valid_x,valid_y,d,hl,ol,config.act_fun,loss_fun,config.epochs,config.eta,config.strat,alpha,config.batch_size)
```
