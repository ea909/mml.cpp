/**
 * A command line MML player. Takes MML text and produces a bandlimited wav
 * file of playback using a square wav. On windows, it can also play the the
 * file back w/ PlaySound.
 *
 * Usage (Windows): mml "song text" [out_file_name]
 * Usage (Other):   mml "song text" out_file_name
 *
 * NOTE: There is some lazy handling of ascii characters still in here so
 * there might be some ways for bad input to crash it.
 *
 * Compile (Windows): cl mml.cpp /link winmm.lib
 * Compile (Other):   clang++ mml.cpp
 */
/*
LICENSE:
Copyright 2014 Eric Alzheimer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <cstdint>
#include <cmath>
#include <cctype>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <fstream>
#include <iostream>

constexpr float     PI = 3.14159265358979323846f;
constexpr int       NUM_OCTAVES = 3;
constexpr int       NOTES_PER_OCTAVE = 12;
constexpr int       NOTE_A_440 = 21;
constexpr int       SAMPLE_RATE = 44100;
constexpr int       TICK_LENGTH = 2700;

constexpr size_t    WAVETABLE_SIZE = 1024; // Must be power of 2

// Amount to right shift u32 to convert into table index:
constexpr uint32_t  WAVETABLE_SHIFT = 22; 

// Mask to get bits below the wavetable index in a u32. This is the fractional
// part of the phase counter
constexpr uint32_t  WAVETABLE_MASK = 0x3FFFFF; 

constexpr size_t    WAVETABLE_NUM_TABLES = 8;
constexpr float     WAVETABLE_BASE_FREQ = 40.0f;
constexpr float     WAVETABLE_CUTOFF_FREQ = 20000.0f;

// Bandlimited wavetables (ie, mipmapped) of a square wave
class SquareWavetable {
    std::array<std::array<float, WAVETABLE_SIZE>, WAVETABLE_NUM_TABLES> data;
    std::array<uint32_t, WAVETABLE_NUM_TABLES> topPhaseRate;
public:
    void Generate(int sampleRate);
    SquareWavetable(int sampleRate) { Generate(sampleRate); }

    // Give a phase rate (in phase increments per sample), return the index of
    // the lowest table that will not alias at that playback speed.
    size_t GetTable(uint32_t phaseRate);

    // phase is a 32 bit fixed point from 0 to 1, spanning the range of the table.
    // Looks up a value from the table selected by the table index using
    // linear interpolation.
    float Lookup(uint32_t phase, size_t table);
};

void SquareWavetable::Generate(int sampleRate) {
    float frequency = WAVETABLE_BASE_FREQ;

    // We start, on the bottom table, with all harmonics from the base freq
    // up. Each subsequent table is used for notes at double the pitch/freq,
    // so there are half as many harmonics. This repeats until you either have
    // enough tables, or the number of them is 1, in which case the final
    // table is a sine wave at the cutoff frequency.
    int maxHarmonics = (int)(WAVETABLE_CUTOFF_FREQ / WAVETABLE_BASE_FREQ);
    for (size_t tableNum = 0; tableNum < WAVETABLE_NUM_TABLES; tableNum++) {
        data[tableNum].fill(0.0f);
        for (int harmonic = 1; harmonic <= maxHarmonics; harmonic++) {
            // Square wave rule: only odd harmonics with inverse proportion decay
            if (!(harmonic & 1)) { continue; }
            float level = 1.0 / harmonic;
            for (size_t i = 0; i < WAVETABLE_SIZE; i++) {
                data[tableNum][i] += 
                    level * sinf(2 * PI * harmonic * (float)i / (float)WAVETABLE_SIZE);
            }
        }

        // Normalize the waveform
        auto max = *std::max_element(data[tableNum].begin(), data[tableNum].end());
        for (auto &elem : data[tableNum]) { elem /= max;}

        topPhaseRate[tableNum] = (uint32_t)(UINT32_MAX * 2 * frequency / sampleRate);
        frequency    *= 2; 
        maxHarmonics /= 2;
        if (maxHarmonics == 0) { maxHarmonics = 1;}
    }
}

size_t SquareWavetable::GetTable(uint32_t phaseRate) {
    return std::distance(topPhaseRate.begin(), 
        std::lower_bound(topPhaseRate.begin(), topPhaseRate.end() - 1, phaseRate));
}

float SquareWavetable::Lookup(uint32_t phase, size_t table) {
    uint32_t left = phase >> WAVETABLE_SHIFT;
    uint32_t right = (phase + WAVETABLE_MASK + 1) >> WAVETABLE_SHIFT;
    float fraction = (float)(phase & WAVETABLE_MASK) / (float)(WAVETABLE_MASK + 1);
    float s1 = data[table][left];
    float s2 = data[table][right];
    return s1 + (s2 - s1) * fraction;
}

#ifdef _WIN32
extern "C" int __stdcall PlaySoundA(const char * pszSound, void *hmod, uint32_t fdwSound);
#define SND_MEMORY          0x0004  /* pszSound points to a memory file */
#endif

const std::array<int, 7> letterToNoteNumber = {
    // a    b   c   d   e   f   g
       9,  11,  0,  2,  4,  5,  7
};

const std::array<int, 10> lengthNumberToTickCount = {
    // 0    1   2   3   4   5   6   7   8   9
       1,   2,  3,  4,  6,  8,  12, 16, 24, 32
};

// MMLPlayer does not produce audio itself. Instead, it reads the MML data and
// produces a sequence of phase rates, one per tick of the song. The phase
// rate indicates the rate per sample to move through a wavetable or similar,
// and a rate of 0 indicates silence.
class MMLPlayer {
    std::array<uint32_t, NUM_OCTAVES * NOTES_PER_OCTAVE> noteToPhaseRate;
    std::vector<char> song;
    int position;
    int octave;

    uint32_t output;
    int tempo;
    int counts;

    char ReadNumber(int min, int max, const char *errorstr);
public:
    MMLPlayer(int sampleRate);
    MMLPlayer(int sampleRate, const char *songstr, int songstrLen) : 
        MMLPlayer(sampleRate) { Load(songstr, songstrLen); }
    void Load(const char *songstr, int songstrLen);
    uint32_t Tick();
    bool IsDone();
};

MMLPlayer::MMLPlayer(int sampleRate) : 
    position(0), octave(1), output(0), tempo(4), counts(0) {
    for (int i = 0; i < noteToPhaseRate.size(); i++) {
        double diff = i - NOTE_A_440;
        double freq = 440.0 * pow(2, diff / 12);
        double rate = freq / sampleRate;
        noteToPhaseRate[i] = (uint32_t)(UINT32_MAX * rate);
    }
}

bool MMLPlayer::IsDone() { return position < 0; }

void MMLPlayer::Load(const char *songstr, int len) {
    song.clear();

    // Convert all of the letters to uppercase
    std::transform(songstr, songstr + len, std::back_inserter(song), [](char c) { 
        return toupper(c);
    });

    song.push_back('\0');

    octave = 1;
    position = 0;
    output = 0;
    counts = 0;
    tempo = 4;
}

char MMLPlayer::ReadNumber(int min, int max, const char *errorstr) {
    char next = song[position++] - '0';

    if (next >= min && next <= max) {
        return next;
    } else {
        throw std::domain_error(errorstr);
    }
}

uint32_t MMLPlayer::Tick() {
    bool done = false;
    int pitch;
    char next, curr;

    // If counts is non-zero, we are still outputting the last note or rest
    // for more ticks:
    if (--counts > 0) { return output; }
    if (position < 0) { return 0; }

    while (!done) {
        switch (curr = song[position++]) {
            case '\0': // End of song
                position = -1;
                output = 0;
                done = true;
                break;
            case '>': // Octave up
                if (octave < NUM_OCTAVES - 1) { octave++; }
                break;
            case '<': // Octave down
                if (octave > 0) { octave--; }
                break;
            case 'O': // Set octave
                octave = ReadNumber(0, NUM_OCTAVES-1, "Invalid O command in song string");
                break;
            case 'T': // Set tempo
                tempo = ReadNumber(0, 9, "Invalid T command in song string");
                break;
            case ' ': case '\n': case '\r': case '\t': // Skip whitespace
                break;
            case 'R': // Rest - output silence
                counts = (tempo + 1) * lengthNumberToTickCount[ReadNumber(0, 9, "Invalid R command in song string")];
                output = 0;
                done = true;
                break;
            case 'A': case 'B': case 'C': case 'D': // Note - output wave at pitch
            case 'E': case 'F': case 'G': 
                pitch = letterToNoteNumber[curr - 'A']; 
                
                next = song[position++];

                // There is an optional sharp or flat symbol after a note
                // name:
                switch (next) {
                    case '#': case '+': pitch++; break;
                    case '-': pitch--; break;
                    default: position--; break; // unconsume - this was the number
                }

                if (pitch >= NUM_OCTAVES * NOTES_PER_OCTAVE) {
                    pitch--;
                } else if (pitch < 0) {
                    pitch++;
                }

                // Set counts to the number of ticks to output the note for:
                counts = (tempo + 1) * lengthNumberToTickCount[ReadNumber(0, 9, "Inavlid count number in note command in song string")];

                // Set output to the phase rate of the note:
                output = noteToPhaseRate[pitch + octave * 12];
                done = true;
                break;
            default:
                throw std::domain_error("Invalid character in song string");
                break;
        }
    }

    return output;
}

// TODO(eric): Check endianess in WriteWaveFile and flip stuff if
// necessary
#pragma pack(push, 1)
struct WAVHeader {
    char chunkId[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2Id[4];
    uint32_t subchunk2Size;
};
#pragma pack(pop)

void BuildWaveHeader(WAVHeader &hdr, int nsamples) {
    hdr.chunkId[0] = 'R'; hdr.chunkId[1] = 'I';
    hdr.chunkId[2] = 'F'; hdr.chunkId[3] = 'F';
    // chunksize is the next field, but is calc'ed below
    hdr.format[0] = 'W'; hdr.format[1] = 'A';
    hdr.format[2] = 'V'; hdr.format[3] = 'E';
    hdr.subchunk1Id[0] = 'f'; hdr.subchunk1Id[1] = 'm';
    hdr.subchunk1Id[2] = 't'; hdr.subchunk1Id[3] = ' ';
    hdr.subchunk1Size = 16;
    hdr.audioFormat = 1;
    hdr.numChannels = 1;
    hdr.sampleRate = SAMPLE_RATE;
    hdr.byteRate = SAMPLE_RATE * 1 * 16 / 8;
    hdr.blockAlign = 1 * 16 / 8;
    hdr.bitsPerSample = 16;
    hdr.subchunk2Id[0] = 'd'; hdr.subchunk2Id[1] = 'a';
    hdr.subchunk2Id[2] = 't'; hdr.subchunk2Id[3] = 'a';
    hdr.subchunk2Size = nsamples * 1 * 16 / 8;
    hdr.chunkSize = 36 + hdr.subchunk2Size;
}

void WriteMonoWaveFile(char *filename, int16_t *data, int nsamples) {
    WAVHeader hdr;
    BuildWaveHeader(hdr, nsamples);
    std::ofstream outfile(filename, std::ios::binary);
    outfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    outfile.write((char*)&hdr, sizeof(hdr));
    outfile.write((char*)data, sizeof(int16_t) * nsamples);
}

#ifdef _WIN32
void PlayMonoWaveData(int16_t *data, int nsamples) {
    std::vector<char> buffer(sizeof(WAVHeader) + sizeof(int16_t) * nsamples);
    WAVHeader *hdr = (WAVHeader*)buffer.data();
    int16_t *wavdata = (int16_t*)(hdr + 1);
    BuildWaveHeader(*hdr, nsamples);
    std::copy(data, data + nsamples, wavdata);
    PlaySoundA(buffer.data(), NULL, SND_MEMORY);
}
#endif

std::vector<int16_t> GenerateSongSquareWave(const char *songstr, int len) {
    SquareWavetable wavetable(SAMPLE_RATE);
    MMLPlayer player(SAMPLE_RATE, songstr, len);
    std::vector<int16_t> data;
    uint32_t phase = 0, phaseRate = 0;
    for (int i = 0; !player.IsDone(); i++) {
        phaseRate = player.Tick();
        auto tableNum = wavetable.GetTable(phaseRate);
        if (phaseRate == 0) {
            std::fill_n(std::back_inserter(data), TICK_LENGTH, 0);
        } else for (int smp = 0; smp < TICK_LENGTH; smp++) {
            float sample = wavetable.Lookup(phase, tableNum);
            data.push_back((int16_t)(16384 * sample));
            phase += phaseRate;
        }
    }
    
    return data;
}

//const char *str = "t4 o0 c8 r8 d8 r8 e4 o1 < f4 g#4 f4";
//const char *str = "t2E5R1E3R0D3R0E3R0E1R0D1R0>G4R1";
const char *str = "t3 o0 c3 g3 o1 c3 g3 o2 c3 g3";
std::string a = "t0E5R1E3R0D3R0E3R0E1R0D1R0>G4R1<";
std::string b = "F3R0F1R0F1R0A3R0F1R0E1R0D1R0D1R0E5R0";
std::string c = "C3R0C1R0C1R0E3R0C1R0>B1<R0C1R0>B1R0A1R0A1B5R0<";
std::string d = "E1R0E1R0E1R0E1R0E1R0E1R0D1R0E1R0E1R0E1R0D1R0>A1R0A1R0B3R1<";
std::string e = ">A1R0B1R0<C1R0D1R0E1R0F1R0E1R0F3R1A3R1B1R0A1R0F3R0E3R0E1R0E4R0";
std::string demosong = a + b + b + c + c + b + c + d + e;

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            printf("Usage: mml \"songtext\" [fname]\n");
            str = demosong.c_str();
        } else {
            str = argv[1];
        }
        
        auto data = GenerateSongSquareWave(str, strlen(str));

        if (argc > 2) {
            WriteMonoWaveFile(argv[2], data.data(), data.size());
        } else {
    #ifdef _WIN32
            PlayMonoWaveData(data.data(), data.size());
    #endif
        }
    } catch (std::domain_error err) {
        std::cout << "Domain Error: " << err.what() << "\n";
    } catch (std::exception err) {
        std::cout << "Error:" << err.what() << "\n";
    }

    return 0;
}


