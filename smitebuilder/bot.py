import discord
from smitebuilder.main import main
import os

client = discord.Client()

with open(os.path.join("..", "bot_info.txt")) as infile:
    datapath = infile.readline().strip()
    token = infile.readline().strip()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!build"):
        args = message.content.split("_")
        god = args[1]
        queue = args[2]

        builds = None
        try:
            builds = main(datapath, queue, god, 15, True, True)

        except KeyError:
            await message.channel.send(
                "Could not find god "
                + god
                + "\nDon't forget to use correct capitalization!",
            )
            return

        except:
            await message.channel.send(
                "SmiteBuilder encountered an error. Please contact L0RDmomo#7288 on Discord and"
                + " detail the command which caused this."
            )

        if builds is None:
            await message.channel.send("No builds found")

        else:
            send_str = ""
            for smitebuild in builds:
                send_str += "core:\n\t" + str(smitebuild.build.core) + "\n"
                send_str += "optional:\n"
                for option in smitebuild.build.optionals:
                    send_str += "\t" + str(option) + "\n"
                send_str += "confidence: " + str(smitebuild.confidence) + "\n"
                send_str += "\n"

            if not send_str:
                await message.channel.send("No builds found")

            else:
                await message.channel.send(send_str)


client.run(token)
