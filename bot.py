import discord
import easy_smitebuilds

client = discord.Client()

with open("bot_info.txt") as infile:
    datapath = infile.readline().strip()
    token = infile.readline().strip()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!build"):
        god = message.content.split(" ")[1]
        builds = easy_smitebuilds.main(datapath, "conquest", god, 15, 0.5, 0.7)

        if builds is None:
            await message.channel.send("An error occurred")

        else:
            for smitebuild in builds:
                await message.channel.send("dt_rank:", smitebuild.dt_rank)
                await message.channel.send("bnb_rank:", smitebuild.bnb_rank)
                await message.channel.send(
                    "core:", smitebuild.build.core,
                )
                await message.channel.send(
                    "optional:", smitebuild.build.optional,
                )


client.run(token)
