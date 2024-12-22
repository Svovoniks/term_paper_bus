from dataclasses import dataclass
from datetime import  timedelta
import random
from typing import Callable
from bisect import bisect_left
from copy import deepcopy
from multiprocessing import Pool
import math

@dataclass
class RelativeTimeFrame:
    starts_after: timedelta
    duration: timedelta

    def get_inverse(self) -> 'RelativeTimeFrame':
        return RelativeTimeFrame(
            starts_after=self.starts_after+self.duration,
            duration=timedelta(days=1)-self.duration
        )

    def get_end(self) -> timedelta:
        return self.starts_after + self.duration

    def get_copy(self) -> 'RelativeTimeFrame':
        return deepcopy(self)

    def contains(self, tf: 'RelativeTimeFrame'):
        return (self.starts_after <= tf.starts_after and tf.get_end() <= self.get_end())

    def to_time_window(self) -> str:
        return f'{(self.starts_after.seconds//3600 + 6)%24:02}:{self.starts_after.seconds%3600//60:02} -\
            {(self.get_end().seconds//3600+6)%24:02}:{self.get_end().seconds%3600//60:02}'

@dataclass
class DriverType:
    type_id: int
    type_name: str
    check_breaks: Callable[['Driver', list[RelativeTimeFrame]], bool]
    work_hour_count: int
    work_time_frame: RelativeTimeFrame
    workdays: list[int]

def cutout(cut_from: list[RelativeTimeFrame], ls: list[RelativeTimeFrame], min_span: timedelta) -> list[RelativeTimeFrame]:
    pieces = cut_from
    for i in ls:
        if len(pieces) == 0:
            return []

        if i.starts_after + i.duration <= pieces[0].starts_after:
            continue

        if i.starts_after >= pieces[-1].starts_after + pieces[-1].duration:
            continue

        lowest = bisect_left(pieces, i.starts_after, key=lambda a: a.starts_after+a.duration)
        highest = bisect_left(pieces, i.starts_after + i.duration, key=lambda a: a.starts_after+a.duration)

        new_pieces = [pieces[i] for i in range(lowest)]
        if lowest < len(pieces):
            new_piece = RelativeTimeFrame(
                starts_after=pieces[lowest].starts_after,
                duration=(i.starts_after - pieces[lowest].starts_after)
            )
            if new_piece.duration >= min_span:
                new_pieces.append(new_piece)

        if highest < len(pieces):
            new_piece = RelativeTimeFrame(
                starts_after=i.starts_after+i.duration,
                duration=(pieces[highest].starts_after+pieces[highest].duration-i.starts_after-i.duration)
            )
            if new_piece.duration >= min_span:
                new_pieces.append(new_piece)

        for idx in range(highest+1, len(pieces)):
            new_pieces.append(pieces[idx])

        pieces = new_pieces
        if len(pieces) == 0:
            return []

    return pieces


class Driver:
    def __init__(self, driver_type: DriverType, work_day_count: int = 21) -> None:
        self.driver_type: DriverType = driver_type
        self.free_time: list[RelativeTimeFrame] = []
        self.rides: list[list[Ride]] = [[] for _ in range(work_day_count)]
        self.break_checking: list[list[RelativeTimeFrame]] = [[driver_type.work_time_frame.get_copy()] for _ in range(work_day_count)]
        for i in range(work_day_count):
            if i in driver_type.workdays:
                tf = driver_type.work_time_frame.get_copy()
                tf.starts_after += timedelta(days=1) * i
                self.free_time.append(tf)
        self.free_time_sum = timedelta(minutes=0)

        for i in self.free_time:
            self.free_time_sum += i.duration

FULL_WORK_HOURS = RelativeTimeFrame(
    starts_after=timedelta(minutes=0),
    duration=timedelta(days=1),
)

def intersection(tf1: RelativeTimeFrame, tf2: RelativeTimeFrame) -> timedelta:
    st = max(tf1.starts_after, tf2.starts_after)
    en  = min(tf1.get_end(), tf2.get_end())
    diff = en - st
    return max(timedelta(minutes=0), diff)


def get_correct_day_idx(driver: 'Driver', day_idx: int, tf: RelativeTimeFrame):
    wf = driver.driver_type.work_time_frame.get_copy()

    wf.starts_after += timedelta(days=day_idx)
    if wf.contains(tf):
        return day_idx

    while wf.starts_after > tf.starts_after:
        wf.starts_after -= timedelta(days=1)
        day_idx -= 1

    while wf.starts_after < tf.starts_after:
        wf.starts_after += timedelta(days=1)
        day_idx += 1

    return day_idx

def check_breaks_8h(driver: 'Driver', free_time: list[RelativeTimeFrame]) -> bool:
    break_time  = RelativeTimeFrame(
        starts_after=driver.driver_type.work_time_frame.starts_after + timedelta(hours=4),
        duration=timedelta(hours=3)
    )

    for i in free_time:
        if intersection(i, break_time) >= timedelta(hours=1):
            return True

    return False

def DRIVER_8_H(start_after: timedelta, workday: int):
    workdays = []

    for i in range(3):
        for j in range(5):
            workdays.append(i*7+((j+workday)%7))

    work_time = RelativeTimeFrame(
        starts_after=start_after,
        duration=timedelta(hours=9),
    )

    return DriverType(
        type_id=0,
        work_hour_count=8,
        type_name='8 hour',
        check_breaks=check_breaks_8h,
        work_time_frame=work_time,
        workdays=workdays,
    )

def check_breaks_12h(driver: 'Driver', free_time: list[RelativeTimeFrame]) -> bool:
    last_break_end = driver.driver_type.work_time_frame.starts_after
    for i in free_time:
        if i.starts_after - last_break_end > timedelta(hours=4):
            return False

        if i.duration > timedelta(minutes=15):
            last_break_end = i.starts_after + i.duration

    return True


def DRIVER_12_H(start_after: timedelta, workday: int):
    workday %= 3
    workdays = [i for i in range(workday, 21, 3)]
    work_time = RelativeTimeFrame(
        starts_after=start_after,
        duration=timedelta(hours=12)
    )

    return  DriverType(
        type_id=1,
        work_hour_count=12,
        type_name='12 hour',
        work_time_frame=work_time,
        check_breaks = check_breaks_12h,
        workdays=workdays,
    )

class Bus:
    def __init__(self, idx: int, work_day_count=21):
        self.bus_idx: int = idx
        tf = FULL_WORK_HOURS.get_copy()
        tf.duration = timedelta(days=work_day_count+1)
        self.free_time: list[RelativeTimeFrame] = [tf]

@dataclass
class Schedule:
    weekday: list[bool]
    weekend: list[bool]
    ride_duration: timedelta

@dataclass
class Genome:
    schedule: Schedule

class Timeline:
    def __init__(self) -> None:
        self.rush = [7, 8, 17, 18]

    def eval_weekday(self, ls: list[RelativeTimeFrame]) -> float:
        bins = [0]*24

        for i in ls:
            mid = i.starts_after + (i.duration / 2)
            bins[mid.seconds//3600] += 1

        score = 0

        for i in range(len(bins)):
            if i in self.rush:
                score += bins[i]*2
            else:
                score += bins[i]
            if bins[i] == 0:
                score -= 10000

        return score

    def eval_weekend(self, ls: list[RelativeTimeFrame]) -> float:
        bins = [0]*24

        for i in ls:
            mid = i.starts_after + (i.duration / 2)
            bins[mid.seconds//3600] += 1

        score = 0

        for i in range(len(bins)):
            score += bins[i]
            if bins[i] == 0:
                score -= 10000

        return score

    def eval_sched(self, sched: Schedule) -> float:
        weekday: list[RelativeTimeFrame] = [RelativeTimeFrame(timedelta(minutes=i), sched.ride_duration) for i in range(len(sched.weekday)) if sched.weekday[i]]
        weekend: list[RelativeTimeFrame] = [RelativeTimeFrame(timedelta(minutes=i), sched.ride_duration) for i in range(len(sched.weekend)) if sched.weekend[i]]

        score = (len(weekend) + len(weekday)) * 7
        score += (self.eval_weekday(weekday) + self.eval_weekend(weekend)) * 1000

        return score




def has_containing(ls: list[RelativeTimeFrame], tf: RelativeTimeFrame) -> bool:
    for i in ls:
        if i.contains(tf):
            return True
    return False

def get_avail_driver(avail_drivers: list[Driver], tf: RelativeTimeFrame, day_idx: int) -> tuple[int, int]:
    tfcp = tf.get_copy()
    tfcp.starts_after += timedelta(days=day_idx)

    drivers_idxs = [(i, avail_drivers[i]) for i in range(len(avail_drivers))]
    drivers_idxs.sort(key=lambda a: a[1].free_time_sum)
    for idx, driver in drivers_idxs:
        if has_containing(driver.free_time, tfcp):
            driver_day = get_correct_day_idx(driver, day_idx, tf)

            cp = driver.break_checking[driver_day].copy()
            cp = cutout(cp, [tf], timedelta(minutes=1))
            if driver.driver_type.check_breaks(driver, cp):
                return idx, driver_day

    return -1, -1

def get_avail_bus(busses: list[Bus], tf: RelativeTimeFrame, day_idx: int) -> int:
    tfcp = tf.get_copy()
    tfcp.starts_after += timedelta(days=day_idx)
    for bus_idx in range(len(busses)):
        if has_containing(busses[bus_idx].free_time, tfcp):
            return bus_idx

    return -1

@dataclass
class Ride:
    driver_id: int
    bus_id: int
    day: int
    tf: RelativeTimeFrame

def book_ride(drivers: list[Driver], busses: list[Bus], ride: RelativeTimeFrame, day_idx: int) -> tuple[bool, Ride | None]:
    bus_idx = get_avail_bus(busses, ride, day_idx)
    if bus_idx == -1:
        return False, None

    driver_idx, driver_day = get_avail_driver(drivers, ride, day_idx)
    if driver_idx == -1:
        return True, None


    tfcp = ride.get_copy()
    tfcp.starts_after += timedelta(days=day_idx)

    drivers[driver_idx].free_time = cutout(drivers[driver_idx].free_time, [tfcp], timedelta(minutes=1))

    drivers[driver_idx].break_checking[driver_day] = cutout(drivers[driver_idx].break_checking[driver_day], [ride], timedelta(minutes=1))
    drivers[driver_idx].free_time_sum -= tfcp.duration

    sm = timedelta()
    for i in drivers[driver_idx].free_time:
        sm += i.duration


    busses[bus_idx].free_time = cutout(busses[bus_idx].free_time, [tfcp], timedelta(minutes=1))
    ride_ = Ride(driver_id=driver_idx, bus_id=bus_idx, day=day_idx, tf=ride)
    drivers[driver_idx].rides[day_idx].append(ride_)

    return True, ride_

def mutate_trips(sched: Schedule, mutation_rate):
    for i in range(len(sched.weekday)):
        if random.random() < mutation_rate:
            sched.weekend[i] = not sched.weekend[i]

    for i in range(len(sched.weekend)):
        if random.random() < mutation_rate:
            sched.weekday[i] = not sched.weekday[i]

def mutate(gen: Genome, mutation_rate: float):
    mutate_trips(gen.schedule, mutation_rate)

def get_child(gen1: Genome, gen2: Genome):
    return Genome(
        schedule=Schedule(
        [random.choice([x, y]) for x, y in zip(gen1.schedule.weekday, gen2.schedule.weekday)],
        [random.choice([x, y]) for x, y in zip(gen1.schedule.weekday, gen2.schedule.weekday)],
        gen1.schedule.ride_duration)
    )

def eval_genome(gen: Genome, bus_count: int) -> float:
    score = 0

    drivers = check_drivers(gen.schedule, bus_count)
    score = Timeline().eval_sched(gen.schedule)

    score -=  len(drivers) * 3000
    return score

def eval_wrapper(tp: tuple[Genome, int]):
    return (tp[0], eval_genome(tp[0], tp[1]))

def eval_pr(gens: list[Genome], bus_count):
    with Pool() as pool:
        data = [(i, bus_count) for i in gens]
        return pool.map(eval_wrapper, data)

def get_driver(ride: RelativeTimeFrame, day_idx: int) ->  DriverType:
    if 0 < int(ride.starts_after.total_seconds())//3600 < 4:
        return DRIVER_8_H(ride.starts_after, day_idx)

    return DRIVER_12_H(ride.starts_after, day_idx)

def check_drivers(schedule: Schedule, bus_count: int) -> list[Driver]:
    weekday: list[RelativeTimeFrame] = [RelativeTimeFrame(timedelta(minutes=i), schedule.ride_duration) for i in range(len(schedule.weekday)) if schedule.weekday[i]]
    weekend: list[RelativeTimeFrame] = [RelativeTimeFrame(timedelta(minutes=i), schedule.ride_duration) for i in range(len(schedule.weekend)) if schedule.weekend[i]]

    drivers: list[Driver] = []
    busses = [Bus(i) for i in range(bus_count)]

    for day_idx in range(21):
        if day_idx%7 == 5 or day_idx%7 == 6:
            sched = weekend
        else:
            sched = weekday

        offset = 0
        avail_drivers = [i for i in drivers if day_idx in i.driver_type.workdays or day_idx-1 in i.driver_type.workdays]

        for sched_idx in range(len(sched)):
            sched_el = sched[sched_idx-offset]

            ride = book_ride(avail_drivers, busses, RelativeTimeFrame(sched_el.starts_after, sched_el.duration), day_idx)

            if not ride[0]:
                sched.pop(sched_idx-offset)
                offset+=1
                continue

            if ride[1] is None:
                driver = Driver(get_driver(sched_el, day_idx))
                drivers.append(driver)
                avail_drivers.append(driver)
                if len(drivers) > 100: return [drivers[0]]*1000
                ride = book_ride(avail_drivers, busses, RelativeTimeFrame(sched_el.starts_after, sched_el.duration), day_idx)

            assert ride[1] is not None

    return drivers

def advance_population(population: list[Genome], trip_duration: timedelta, bus_count: int) -> tuple[list[Genome], Genome, float, float]:
    graded = eval_pr(population, bus_count)
    graded = sorted(graded, key=lambda a: a[1], reverse=True)

    best = graded[0]

    retain = 0.1

    retain_length = int(len(graded) * retain)
    parents = [i[0] for i in graded[:retain_length]]

    random_select = 0.2
    for i in graded[retain_length:]:
        if random.random() < random_select:
            parents.append(i[0])

    next_generation: list[Genome] = []
    mutation_rate = 0.000001

    while len(next_generation) < len(population):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        if parent1 != parent2:
            child = get_child(deepcopy(parent1), deepcopy(parent2))
            mutate(child, mutation_rate)
            next_generation.append(child)

    sm = 0
    for i in graded:
        sm += i[1]

    return next_generation, best[0], best[1], sm/len(graded)

def init_population(size: int, trip_duration: timedelta) -> list[Genome]:
    ls: list[Genome] = []

    for _ in range(size):
        sched = Schedule(
            [random.random() < 0.01 for _ in range(24*60)],
            [random.random() < 0.01 for _ in range(24*60)],
            trip_duration,
        )

        ls.append(Genome(schedule=sched))

    return ls


class ScheduleDay:
    def __init__(self, day_idx: int) -> None:
        self.day_idx = day_idx
        self.rides: list[Ride] = []

class ScheduleWeek:
    def __init__(self, week_idx: int) -> None:
        self.week_idx = week_idx
        self.days: list[ScheduleDay] = [ScheduleDay(i) for i in range(7)]

class ScheduleTable:
    def __init__(self) -> None:
        self.weeks: list[ScheduleWeek] = [ScheduleWeek(i) for i in range(3)]

def write_drivers_to_markdown(drivers: list[DriverType], filename: str):
    with open(filename, "w") as file:
        for week_idx in range(3):
            file.write(f"### Week {week_idx + 1}\n\n")
            file.write("| Driver ID | Driver Type   | Mon | Tue | Wed | Thu | Fri | Sat | Sun |\n")
            file.write("|-----------|---------------|-----|-----|-----|-----|-----|-----|-----|\n")

            for driver_idx, driver in enumerate(drivers):
                st = f'| {driver_idx} | {driver.type_name}'
                for day_idx in range(7):
                    st += '| '
                    if week_idx * 7 + day_idx in driver.workdays:
                        st += driver.work_time_frame.to_time_window()

                st += ' |'

                file.write(st)

                file.write("\n")

def write_schedule_to_markdown(table: ScheduleTable, filename: str):
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    with open(filename, "w") as file:
        for week in table.weeks:
            file.write(f"### Week {week.week_idx + 1}\n\n")
            file.write("| Day       | Time      | Bus ID | Driver ID |\n")
            file.write("|-----------|-----------|--------|-----------|\n")

            for day in week.days:
                for ride in sorted(day.rides, key=lambda a: a.tf.starts_after):
                    file.write(f"| {days_of_week[day.day_idx]} | {ride.tf.to_time_window()} | {ride.bus_id} | {ride.driver_id} |\n")

            file.write("\n")

def write_schedule_gen(gen: Genome, bus_count: int):
    drivers = check_drivers(gen.schedule, bus_count)
    table = ScheduleTable()

    for driver_idx in range(len(drivers)):
        for idx, rides in enumerate(drivers[driver_idx].rides):
            for rd in rides:
                rd.driver_id = driver_idx
            table.weeks[idx//7].days[idx%7].rides.extend(rides)

    write_schedule_to_markdown(table, 'time_table_gen.md ')
    write_drivers_to_markdown([i.driver_type for i in drivers], 'drivers_gen.md ')


def write_schedule_brute(drivers: list[Driver]):
    table = ScheduleTable()

    for driver_idx in range(len(drivers)):
        for idx, rides in enumerate(drivers[driver_idx].rides):
            for rd in rides:
                rd.driver_id = driver_idx
            table.weeks[idx//7].days[idx%7].rides.extend(rides)

    write_schedule_to_markdown(table, 'time_table_brute.md ')
    write_drivers_to_markdown([i.driver_type for i in drivers], 'drivers_brute.md ')

def get_schedule(rush: list[int], default_gap: timedelta, rush_gap: timedelta, trip_time: timedelta) -> list[RelativeTimeFrame]:
    cur_time = timedelta(minutes=0)
    end_time = timedelta(days=1)

    sched = []

    while cur_time < end_time:
        sched.append(RelativeTimeFrame(cur_time, trip_time))
        if int(cur_time.total_seconds())//3600+6 in rush:
            cur_time += rush_gap
        else:
            cur_time += default_gap

    return sched
