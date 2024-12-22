from datetime import timedelta

from lib import (
    write_schedule_brute,
    book_ride,
    get_schedule,
    Driver,
    get_driver,
    RelativeTimeFrame,
    Bus
)

if __name__ == '__main__':
    print('Enter the time it takes to finish a route (in minutes)')
    trip_time = int(input(' > '))

    print('Enter the number of busses')
    bus_count = int(input(' > '))

    weekend = get_schedule([], timedelta(minutes=30), timedelta(minutes=10), timedelta(minutes=trip_time) + timedelta(minutes=10))
    weekday = get_schedule([7, 8, 17, 18], timedelta(minutes=30), timedelta(minutes=10), timedelta(minutes=trip_time) + timedelta(minutes=10))

    drivers = []
    busses = [Bus(i) for i in range(bus_count)]

    for day_idx in range(21):
        if day_idx%7 == 5 or day_idx%7 == 6:
            sched = weekend
        else:
            sched = weekday

        offset = 0
        for sched_idx in range(len(sched)):
            sched_el = sched[sched_idx-offset]

            ride = book_ride(drivers, busses, RelativeTimeFrame(sched_el.starts_after, sched_el.duration), day_idx)

            if not ride[0]:
                sched.pop(sched_idx-offset)
                offset+=1
                continue

            if ride[1] is None:
                driver = get_driver(sched_el, day_idx)
                drivers.append(Driver(driver))
                ride = book_ride(drivers, busses, RelativeTimeFrame(sched_el.starts_after, sched_el.duration), day_idx)

            assert ride[1] is not None

    print('driver count:', len(drivers))
    print('weekday rides:', len(weekday))
    print('weekend rides', len(weekend))
    write_schedule_brute(drivers)
