# TorchHelper
A torch helper for control log、train process（train/test/eval/checkpoint/save）


## Frame - A Model Frame for All
### Trainer
Trainer致力于简化模型的训练、验证、测试、断点、保存等一系列过程只要遵循Trainer的构造逻辑，即可迅速完成一整个训练流程

`Trainer`内部依托torch实现了较为通用的train、eval、test逻辑，并且提供了准确率统计等方法

- `Trainer`通过包内的另外一个类`Saver`实现了模型的保存和断点保存，使得训练过程中无需手动保存，只需要在类内指定模型和优化器和Saver，即可在训练过程中自动保存

- `Trainer`内部维护了`eidx`和`global_step`两个变量，用来帮助类内和类外获取训练的次数信息

- 与`Trainer`内部重写了`__getattr__`方法，配套的`TrainerParam`需要同时被创建导入，`TrainerParam`类内部的所有参数都可以在类内由self直接访问

- 可以通过`self.log_handler`方法方便的控制对哪些方法输出日志，其参数及其详细使用方法参考Logger



```python
from torchhelper.frame.trainer import SupervisedTrainer

class SingleModelTrainer(SupervisedTrainer):
    _loss_dict = {
        "mse": nn.MSELoss(),
        "cross_entropy": nn.CrossEntropyLoss(),
    }

    def __init__(self, params, model: nn.Module, optimizer, loss, sup_dataloader, eval_dataloader, test_dataloader,
                 device, logged=True):
        super().__init__(params, sup_dataloader, eval_dataloader, test_dataloader, device)
        self.model = model
        self.optimizer = optimizer
        # TODO 分 str、fn 两类做判断
        if isinstance(loss, str):
            assert loss in SingleModelTrainer._loss_dict, "can't reco the loss"
            self.lossfn = SingleModelTrainer._loss_dict[loss]
        elif callable(loss):
            self.lossfn = loss
        else:
            assert False, "loss must be str or callable, but get %s" % loss

        self.regist_model_name(self.model.__class__.__name__)
        self.regist_model_and_optim(model=self.model, optim=self.optimizer)

        if logged:
            self.logger = Logger()
            self.log_handle(self.logger, self.train, prefix="Train model: ")
            self.log_handle(self.logger, self.test, prefix="Test  model: ")
            self.log_handle(self.logger, self.eval, prefix="Eval  model: ")
            self.log_handle(self.logger, self.save_checkpoint, prefix="Save Checkpoint: ")
            self.log_handle(self.logger, self.save_model, prefix="Save model: ")
            self.log_handle(self.logger, self.train_epoch, inline=True, prefix="Save model: ", cacu_step=True)

    def predict(self, x):
        return self.model(x)

    def train_epoch(self, eidx):
        for idx, (xs, labels) in enumerate(self.sup_dataloader):
            logits = self.model(xs)
            loss = self.lossfn(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


model = get_model(...)
params = TrainerParam()
optimizer = ...
train_dataloader = ...
eval_dataloader = ...
test_dataloader = ...
trainer = SingleModelTrainer(
            params,model,optimizer,loss = "mse",
            train_dataloader,eval_dataloader,test_dataloader,
            logged = True)

trainer.train()
```

### Saver
Saver用于方便的保存模型和断点，和Trainer集成，也可以单独使用
```python
from torchhelper.frame.saver import Saver
saver = Saver("./models/",max_to_keep=3)
saver.save(model.state_dict(),"model_name")
model.load_state_dict(saver.load("model_name"))
saver.checkpoint(epoch=10,ckpt_dict={
    "model_name":model.state_dict(),
    "...":...
})
ckpt_dict = saver.restore(pointer=-1) # lastest checkpoint

model.load_state_dict(ckpt_dict["model_name"])
epoch = ckpt_dict["epoch"]
```


### Logger
Logger用于更快的输出日志，同样和Trainer集成，也可以单独使用。

- Logger实例化后，可以通过装饰器 @logger.logger()来对方法进行装饰，被装饰方法可以返回一个LogParam()对象，该对象的所有变量都会被logger格式化输出。

- Logger方法可以控制是否在一行内输出（仅在循环逻辑中有效，不同方法仍然会换行），以及是否计算函数执行时间，具体有哪些参数参考该方法内部注释

- Trainer内部实现了log_handle方法，该方法接受的第一个参数是一个实例化的logger，之后的参数为logger.logger装饰器的参数

- 如果有后期对某方法进行包装的需求，则使用logger.wrap方法，该方法的第一个参数是被装饰方法，之后的参数和装饰器参数相同

```python
from torchhelper.frame.logger import Logger,LogParam

logger = Logger()

@logger.logger(inline=True, not_in_class=True)
def cacu(eidx):
    for i in range(10000):
        meter = LogParam()
        meter.loss = i
        meter.accr = 0.64
        meterk = meter.k
        meter.percent(meterk.accr)
        yield meter

for i in range(100):
    cacu(i)

```

## tune
### Branch
Branch是一个模仿自TensorFlow的静态图逻辑，用于控制函数执行流程的类，可以用来灵活的在复杂逻辑中控制只执行部份函数得到部份结果。

下面的例子用来掩饰在一个小的加法和乘法逻辑中，如果只要加或者乘的函数被执行，所需要执行的步骤

代码并不是很pythonic，虽然有一些自认为巧妙的构造，简化了代码逻辑，但实际上并不易懂。

```python
def add(a, b):
    print("add")
    return a + b

def multi(a, b):
    print("multi")
    return a * b

def swap(a, b):
    print("swap")
    return b, a

p = PseudoVar()
br = Branch()

br.add_node = add, (p.a, p.b), (p.c) # equal to add, br(by = (p.a,p.b),to = (p.c), force_dis=False,replace=None)
br.multi = multi, (p.a, p.c), (p.d)
br.swap = swap, (p.a, p.b), (p.sa, p.sb)
res = br.run_for(p.d, p.sa, a=1, b=2)

print(res,type(res))
print(res.d)

# or
br.add_needs(p.d,p.sa)
br.run(a=1,b=2)

# 如果不用该函数，那么执行逻辑会变成类似下面的
# 代码行数并没有少或者多太多
# 但是如果该执行逻辑很复杂，需要很多个if语句来重复判断某个arg时，上面的branch方式就会很方便的自动执行流程
cacu_add = True
cacu_multi = True

c = 0
if cacu_add:
    c = add(a,b)

if cacu_multi:
    d = multi(a,c)
```