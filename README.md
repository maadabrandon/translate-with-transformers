# PLAN OF ACTION

The generally recommended approach to a problem like this would be to download a pre-trained RNN or transformer from say, Hugging Face, and then fine-tune it to my particular use case. I have no doubt that this would be a (relatively) fast method of solving the problem, and that it is likely to produce good results. 

I would be remiss however, if I didn't admit that I like the idea of building custom models from scratch, because:

    - I want to know how transformers actually work in as much detail as I can.
    - I want to learn how to debug problems that will inevitably occur when you build them from scratch.
    - I want to see and write as much Pytorch code as I possibly can (because I really like Pytorch), and this is an excuse to write some more.

Because of this, I will aim to accomplish both things:

    - fine-tune a pretrained model (something I haven't done before).
    - build a transformer of my own from scratch (following the original architecture proposed in "Attention is All You Need"). 

I'm under no illusions that any model I will build from scratch will beat any of the well known pre-trained options that are available. In fact, I am almost certain to run into an avalanche of problems that I will have to debug. Hopefully, I will come out the other side with some amount of success.

Following the sucessful creation of both models, I could look at any (more promising) transformer architectures that have been created since "Attention is All You Need" was published, and
    - learn how to build them from scratch as well
    - fine-tune any existing pre-trained implementations.

Hopefully, training these various models will not prove to be prohibitively expensive (in terms of some combination of time, data, and money).