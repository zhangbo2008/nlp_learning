# HMM

基于bigram, trigram实现的HMM， 支持viterbi解码输出更高效！
https://mp.weixin.qq.com/s?__biz=MzIwNDM1NjUzMA==&mid=2247483662&idx=1&sn=cf463dde9af1844a3fd1e3e4fec26f5c&chksm=96c02fd3a1b7a6c5cfabe53efbff54af33cd2f61d13064645fbff92ce1b024d82acb2375d9b0#rd




sentence = x1 x2 ... xn

tag = y1 y2 ... yn



下面以trigram为例。

trigram 隐马尔可夫模型下有如下定义:



p(X1=x1, X2=x2, Y1=y1, Y2=y2) 

= Π p(Yi=yi | Yi-2=yi-2, Yi-1=yi-1) * Π p( Xi=xi | Yi=yi )
#上面的第一项是= Π p(Yi=yi) 因为马尔科夫他只跟前2个状态相关.就是全概率公式的书写而已.
就是让P这个函数趋近于1即可.



具体计算: 核心!!!!!!!!!!!!!!!!!

p(X1=x1, X2=x2, ... , Xn=xn, Y1=y1, Y2=y2, ... , Yn+1=yn+1) 
记做2个函数一个q,一个e
= Π q(Yi=yi | Yi-2=yi-2, Yi-1=yi-1) * Π e( Xi=xi | Yi=yi )
这里面为什么是假设Y满足马尔科夫,而不是x满足马尔科夫呢?????
因为输入的东西是已经给定的.不能修改,结果可以随便赋予一些假设来简化计算.


那么 q 和 e 如何算呢？



接下来来定义几个量：

1. 词性标记的集合V

2. 对于u, v, w ∈ V 有


就是结果是连续的u,v,w   马尔科夫可以看做一个简单的时间序列
c (u, v, w) 为三元组(u, v, w) 出现的次数

c (u, v) 为二元组(u, v) 出现的次数

c (w->x) 为输入词x被标记为w的次数

c (w) 为标记w出现的次数



接下来有



q(w|u, v) = c (u, v, w) / c(u, v)

e(x|w) = c(w->x) / c(w)



下面读viterbi方法. 就是这里面的pdf论文
























