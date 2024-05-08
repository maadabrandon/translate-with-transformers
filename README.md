# PLAN OF ACTION

The generally recommended approach to a problem like this would be to download a pre-trained RNN or transformer from say, Hugging Face, and then fine-tune it to my particular use case. I have no doubt that this would be a (relatively) fast method of solving the problem, and that it is likely to produce good results. 

I would be remiss however, if I didn't admit that I like the idea of building custom models from scratch, because:

    - I want to know how transformers actually work in as much detail as I can.
    - I want to learn how to debug problems that will inevitably occur when you build them from scratch.
    - I want to see and write as much Pytorch code as I possibly can (because I really like Pytorch), and this is an excuse to write some more.

Because of this, my first task will be to build a transformer from scratch (following the original architecture proposed in "Attention is All You Need"). I'm under no illusions that it will beat any of the well known pre-trained options that are available. In fact, I am almost certain to run into an avalanche of problems that I will have to debug. However, this is an itch that I must scratch. After the model is completed (and successfully debugged), I will attempt to train it if I find that doing so on my dataset is actually feasible (in terms of time and money). 

After doing this, I will:

    - look at any new (more promising) transformer architectures that have been created and learn how to build them from scratch as well, before once again attempting to traini them.

Regardless of whether training any of these models from scratch proves to be prohibitively expensive (which will very likely be the case because of some combination of limited time, data, and money), I will also make use of pretrained models. After all, this will be my first experience when it comes to fine tuning pre-trained models.