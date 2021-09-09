import salabim as sim

# class Car(sim.Component):
#     def process(self):
#         while True:
#             yield self.hold(1)

# env = sim.Environment(trace=True)
# Car()
# env.run(till=5)

class CustomerGenerator(sim.Component):
    def process(self):
        while True:
            Customer()
            yield self.hold(sim.Uniform(5,15).sample())

class Customer(sim.Component):
    def process(self):
        self.enter(waitingLine)
        if clerk.ispassive():
            clerk.activate()
        yield self.passivate()

class Clerk(sim.Component):
    def process(self):
        while True:
            while len(waitingLine) == 0:
                yield self.passivate()
            self.customer = waitingLine.pop()
            yield self.hold(30)
            self.customer.activate()

env = sim.Environment(trace=True)

CustomerGenerator()
clerk = Clerk()
waitingLine = sim.Queue("waitingLine")

env.run(till=50)

print()
waitingLine.print_statistics()