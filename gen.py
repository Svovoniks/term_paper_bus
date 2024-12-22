import math
from datetime import timedelta

from lib import (
    write_schedule_gen,
    init_population,
    advance_population,
    check_drivers,
)


if __name__ == '__main__':
    print('Enter the time it takes to finish a route (in minutes)')
    trip_time = int(input(' > '))
    print('Enter the number of busses')
    bus_count = int(input(' > '))

    size = 100
    trip_duration = timedelta(minutes=trip_time) + timedelta(minutes=10)

    population = init_population(size, trip_duration)

    last_changed = 0
    best = None
    best_score = -math.inf
    last_gen = 0

    for i in range(2000):
        population, new_best, score, avg = advance_population(population, trip_duration, bus_count)

        if score > best_score:
            last_changed = i
            best = new_best
            best_score = score
            write_schedule_gen(best, bus_count)

        assert best is not None
        drivers = check_drivers(best.schedule, bus_count)
        print(f'\
gen: {i+1:03}:  \
best: {score:.2f}  \
cur_best:{best_score:.2f}  \
drivers: {len(check_drivers(new_best.schedule, bus_count))}  \
weekend: {sum(new_best.schedule.weekend)}  \
weekday: {sum(new_best.schedule.weekday)}  \
best_drivers: {len(drivers)}  \
best_weekday: {sum(best.schedule.weekday)}  \
best_weekend: {sum(best.schedule.weekend)}  \
avg: {avg}')

        if last_changed + 50 == i:
            break

        last_gen = i

    print('driver count:', len(check_drivers(best.schedule, bus_count)))
    print('weekday rides:', sum(best.schedule.weekday))
    print('weekend rides', sum(best.schedule.weekend))

