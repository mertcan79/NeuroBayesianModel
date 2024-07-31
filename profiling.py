import cProfile
import pstats
import io
import json
from main import main


def profile_main():
    pr = cProfile.Profile()
    pr.enable()
    main()  # Call your main function
    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    profile_main()
