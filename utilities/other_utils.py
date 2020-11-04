import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')  # make sure to execute this before importing the WML client. Otherwise it wont work

def info(*args):
  """ logs the arguments given 
  this function enables a timestamp for every execution; this is a replacement for `print`"""
  infoTxt = args[0]
  for arg in args[1:]:
      infoTxt += ' ' + str(arg)
  logging.info(infoTxt)


def printout(statement: str, flag: str = "info"):
    """
    nicely formatted printout
    :param statement:
    :param flag:
    :return:
    """
    flags = ["info", "error", "reminder"]

    if flag not in flags:
        print("Flag not in pre-defined flags: defaults to 'info'")
        flag = "info"

    print(f"[{flag.upper()}]", statement)
