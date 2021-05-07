from environments.factory.base_factory import BaseFactory


class SimpleFactory(BaseFactory):
    def __init__(self, *args, max_dirt=5, **kwargs):
        super(SimpleFactory, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    factory = SimpleFactory(n_agents=1, max_dirt=2)
    print(factory.state)
    state, r, done, _ = factory.step(0)
    print(state)