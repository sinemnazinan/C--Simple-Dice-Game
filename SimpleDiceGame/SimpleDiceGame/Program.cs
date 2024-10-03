using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace SimpleDiceGame
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int RollCount = 0;
            bool IsCountValid = false;
            //Random sınıfından random adında bir nesne oluşturuyorsun.
            Random RandomNum = new Random();
            int Dice1 = 0;
            int Dice2 = 0;
            Console.WriteLine("Welcome to the Simple Dice Game!");

            while (!IsCountValid)
            {
                Console.WriteLine("How many times do you want to roll the dice");

                if (int.TryParse(Console.ReadLine(), out RollCount) == false)
                {
                    Console.WriteLine("Please enter a valid value");
                }
                else
                {
                    IsCountValid = true;
                }
            }
            for (int i = 0; i < RollCount; i++)
            {
                Console.WriteLine("Press any key to roll the dice");
                Console.ReadKey();
                //Oluşturduğun nesneye, Next() metodu ile rastgele sayı
                //üretme işlemi yaptırıyorsun. Bu rastgele sayıyı ise Dice1 değişkenine atıyorsun.
                Dice1 = RandomNum.Next(1, 7);
                Console.WriteLine("You rolled; " + Dice1);

                System.Threading.Thread.Sleep(1000);

                Dice2 = RandomNum.Next(1, 7);
                Console.WriteLine("Enemy rolled; " + Dice2);

                if (Dice1 == 1)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|   |");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|   |");
                    Console.WriteLine("-----");
                }
                if (Dice1 == 2)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|*  |");
                    Console.WriteLine("|   |");
                    Console.WriteLine("|  *|");
                    Console.WriteLine("-----");
                }
                if (Dice1 == 3)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|*  |");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|  *|");
                    Console.WriteLine("-----");
                }
                if (Dice1 == 4)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|   |");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }
                if (Dice1 == 5)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }
                if (Dice1 == 6)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }
                if (Dice2 == 1)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|   |");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|   |");
                    Console.WriteLine("-----");
                }
                if (Dice2 == 2)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|*  |");
                    Console.WriteLine("|   |");
                    Console.WriteLine("|  *|");
                    Console.WriteLine("-----");
                }
                if (Dice2 == 3)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|*  |");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|  *|");
                    Console.WriteLine("-----");
                }
                if (Dice2 == 4)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|   |");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }
                if (Dice2 == 5)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("| * |");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }

                if (Dice2 == 6)
                {
                    Console.WriteLine("-----");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("|* *|");
                    Console.WriteLine("-----");
                }


                Console.ReadLine();
            }
        }
    }
}

