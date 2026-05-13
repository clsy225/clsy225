# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## This Machine's Notes

### Safe apt workflow on this Raspberry Pi

Do **not** run a plain update/upgrade flow on this setup when system updates are involved.
Use this sequence instead:

```bash
sudo apt-mark hold raspberrypi-bootloader
sudo apt-get update
sudo apt-get upgrade
```

Reason: the user reports that skipping the bootloader hold can break this Raspberry Pi setup.

### Display reference

- 3.5inch RPi Display (MPI3501) docs: <http://www.lcdwiki.com/zh/3.5inch_RPi_Display>
- Notes: SPI display, ILI9486 LCD + XPT2046 touch, Raspberry Pi 5 supported, driver via `LCD-show` / `LCD35-show`.

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

## Related

- [Agent workspace](/concepts/agent-workspace)
